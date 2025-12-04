# Histogram Building

## Overview

Histogram building is the core operation in histogram-based tree training. For each node,
we aggregate gradient statistics (g, h) by bin index, producing a histogram that can be
scanned to find optimal splits.

## The Key Insight

**Traditional approach**: For each potential split point, sort and partition data
- Complexity: O(n × d × log n) per tree
- Memory: Need sorted indices per feature

**Histogram approach**: Aggregate gradients into bins, scan bins
- Complexity: O(n × d) for histogram + O(bins × d) for split finding
- Memory: O(bins × d) for histogram
- When n >> bins (typical: n=millions, bins=256), this is a massive win

## Histogram Structure

A histogram is an array of gradient pairs (g, h) indexed by bin:

```text
hist[bin] = Σ {(g_i, h_i) : row i has feature value in bin}
```

**Example**:
```text
Feature 0, 3 bins:
  Rows in bin 0: {0, 3, 7}  → g=[0.1, 0.2, -0.1], h=[1,1,1]
  Rows in bin 1: {1, 4}     → g=[0.3, -0.2], h=[1,1]
  Rows in bin 2: {2, 5, 6}  → g=[0.0, 0.1, 0.2], h=[1,1,1]

Histogram:
  hist[0] = (0.2, 3.0)   // sum of g=0.2, sum of h=3.0
  hist[1] = (0.1, 2.0)
  hist[2] = (0.3, 3.0)
```

## XGBoost Implementation

### Data Structures

```cpp
// Gradient pair with extended precision for accumulation
struct GradientPairPrecise {
  double grad;
  double hess;
};

// Histogram for one node = array of gradient pairs, one per bin
using GHistRow = Span<GradientPairPrecise>;

// Collection manages histograms for multiple nodes
class HistCollection {
  std::vector<std::vector<GradientPairPrecise>> data_;
  std::vector<size_t> row_ptr_;  // Maps node_id → histogram index
  uint32_t nbins_;               // Total bins across all features
};
```

### Building Algorithm

**Row-wise (default for cache-friendly access)**:

```cpp
// Simplified from hist_util.cc: RowsWiseBuildHistKernel
for (row_idx in rows_in_node) {
  g, h = gradients[row_idx];
  for (col_idx in 0..n_features) {
    bin_idx = gmat.index[row_idx * n_features + col_idx];
    hist[bin_idx].grad += g;
    hist[bin_idx].hess += h;
  }
}
```

**Column-wise (when histogram doesn't fit in L2)**:

```cpp
// Better for wide data with large histograms
for (col_idx in 0..n_features) {
  for (row_idx in rows_in_node) {
    bin_idx = gmat.index[row_idx * n_features + col_idx];
    g, h = gradients[row_idx];
    hist[bin_idx].grad += g;
    hist[bin_idx].hess += h;
  }
}
```

### XGBoost Decision Logic

From `hist_util.cc`:

```cpp
// Use column-wise when histogram doesn't fit in L2 cache
constexpr double kAdhocL2Size = 1024 * 1024 * 0.8;  // ~800KB
bool hist_fit_to_l2 = kAdhocL2Size > 2 * sizeof(float) * n_bins;
bool read_by_column = !hist_fit_to_l2 && !any_missing;
```

## Parallel Histogram Building

XGBoost uses a sophisticated parallel strategy with thread-local histograms.

### The Problem

Multiple threads updating the same histogram bin creates race conditions.

### The Solution: Thread-Local + Merge

```text
┌─────────────────────────────────────────────────────────────┐
│  1. Partition rows across threads                           │
│                                                             │
│  Thread 0: rows [0, 1000)     → local_hist_0                │
│  Thread 1: rows [1000, 2000)  → local_hist_1                │
│  Thread 2: rows [2000, 3000)  → local_hist_2                │
│  ...                                                        │
│                                                             │
│  2. Each thread builds into its local histogram              │
│     (no synchronization needed)                              │
│                                                             │
│  3. Merge phase: reduce local histograms into final          │
│     ParallelFor(bins, [&](bin) {                             │
│       for (tid in threads) {                                 │
│         final_hist[bin] += local_hist[tid][bin];            │
│       }                                                      │
│     });                                                      │
└─────────────────────────────────────────────────────────────┘
```

### ParallelGHistBuilder

XGBoost's `ParallelGHistBuilder` manages thread-local histograms:

```cpp
class ParallelGHistBuilder {
  HistCollection hist_buffer_;        // Thread-local histograms
  std::vector<GHistRow> targeted_hists_;  // Final output histograms
  std::map<pair<tid, nid>, int> tid_nid_to_hist_;  // Mapping
  
  // Get histogram for thread to write to
  GHistRow GetInitializedHist(size_t tid, size_t nid);
  
  // Merge thread-local histograms
  void ReduceHist(size_t nid, size_t begin, size_t end);
};
```

## Histogram Subtraction Trick

**Key optimization**: For a parent node split into left and right children:

```text
parent_hist = left_hist + right_hist
```

Therefore:

```text
right_hist = parent_hist - left_hist  (or vice versa)
```

We only need to build histogram for **one** child (typically the smaller one),
then derive the other by subtraction.

### Implementation

From `histogram.cc`:

```cpp
void SubtractionHist(GHistRow dst, GHistRow src1, GHistRow src2,
                     size_t begin, size_t end) {
  for (size_t i = begin; i < end; ++i) {
    dst[i].grad = src1[i].grad - src2[i].grad;
    dst[i].hess = src1[i].hess - src2[i].hess;
  }
}
```

### Node Assignment Logic

Choose which child to build vs subtract:

```cpp
void AssignNodes(tree, candidates, nodes_to_build, nodes_to_sub) {
  for (candidate : candidates) {
    left_nidx = tree.LeftChild(candidate.nid);
    right_nidx = tree.RightChild(candidate.nid);
    
    // Build histogram for child with fewer samples
    if (candidate.split.right_sum.hess < candidate.split.left_sum.hess) {
      nodes_to_build.push(right_nidx);
      nodes_to_sub.push(left_nidx);
    } else {
      nodes_to_build.push(left_nidx);
      nodes_to_sub.push(right_nidx);
    }
  }
}
```

## Prefetching

For row-wise building, XGBoost prefetches upcoming gradient pairs and bin indices:

```cpp
// Prefetch data for upcoming rows
if (do_prefetch) {
  size_t prefetch_row = rid[i + kPrefetchOffset];
  PREFETCH_READ_T0(p_gpair + 2 * prefetch_row);  // Gradient pair
  for (j in prefetch_range) {
    PREFETCH_READ_T0(gradient_index + j);        // Bin indices
  }
}
```

This hides memory latency when row data isn't in cache.

## Multi-Node Histogram Building

XGBoost processes multiple nodes at the same depth in parallel:

```cpp
// blocked space: iterate over (node, row_block) pairs
common::ParallelFor2d(space, n_threads, [&](node_in_set, row_range) {
  auto tid = omp_get_thread_num();
  auto nidx = nodes[node_in_set].nid;
  auto hist = buffer.GetInitializedHist(tid, node_in_set);
  auto rows = row_set_collection[nidx];
  
  BuildHist(gpair, rows[row_range], gmat, hist);
});
```

## Performance Considerations

### Memory Bandwidth

Histogram building is memory-bound:

- Read gradients: 8 bytes per row (2 × f32)
- Read bin indices: 1-4 bytes per feature per row
- Write histogram: Random access, but cached after first pass

### Cache Utilization

- **Row-wise**: Sequential gradient access, random histogram access
- **Column-wise**: Random gradient access, sequential histogram access
- Choose based on histogram size vs L2 cache

### SIMD Potential

Current XGBoost uses scalar accumulation. Potential optimizations:

- SIMD horizontal add for bin index compression/decompression
- AVX gather for non-sequential gradient access
- Not trivial due to indirect histogram indexing

## Considerations for booste-rs

### What We Need

1. **Histogram storage**: Per-node, per-feature gradient sums
2. **Build function**: Aggregate gradients by bin
3. **Subtraction function**: Derive child from parent - sibling
4. **Parallel building**: Thread-local histograms + merge

### Potential Simplifications

1. **Start with row-wise only**: Simpler, good for most cases
2. **Skip prefetching initially**: Add later if profiling shows benefit
3. **Fixed f64 precision**: Simpler than template dispatch

### Potential Improvements

1. **Better cache blocking**: Tune block sizes for Rust/LLVM
2. **SIMD reduction**: Vectorized merge of thread-local histograms
3. **Streaming builds**: Process data in cache-sized chunks

## Source Code References

| Component | XGBoost Source |
|-----------|----------------|
| BuildHist | `src/common/hist_util.cc` |
| ParallelGHistBuilder | `src/common/hist_util.h` |
| HistogramBuilder | `src/tree/hist/histogram.h` |
| SubtractionHist | `src/common/hist_util.cc` |
| AssignNodes | `src/tree/hist/histogram.cc` |
