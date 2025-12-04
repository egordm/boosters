# XGBoost Training Optimizations

## Overview

XGBoost achieves high performance through several algorithmic and implementation
optimizations. This document catalogs the key optimizations we should consider
for booste-rs.

## Algorithmic Optimizations

### 1. Histogram Subtraction Trick

**The Problem**: Building histograms is O(n × bins) per node.

**The Solution**: For sibling nodes, only build histogram for smaller child.

```text
parent histogram = left histogram + right histogram
⟹ sibling histogram = parent histogram - built histogram
```

**Impact**: Reduces histogram builds from 2n-1 to ~1.5n per level.

```cpp
// In XGBoost
void BuildHist(GradientPairPrecise const* gpair,
               const RowSetCollection::Elem& row_indices,
               GHistIndexMatrix const& gmat,
               GHistRow hist) {
  // Build histogram for smaller child
}

// Subtract to get sibling
void SubtractionTrick(GHistRow sibling, GHistRow parent, GHistRow smaller) {
  for (size_t i = 0; i < n_bins; i++) {
    sibling[i] = parent[i] - smaller[i];
  }
}
```

**booste-rs opportunity**: Easy to implement, ~30% speedup on histogram builds.

### 2. Quantile Sketching

**The Problem**: Continuous features need to be binned. Exact quantiles require
sorting O(n log n) per feature.

**The Solution**: Use Greenwald-Khanna sketch for approximate quantiles.

```text
Sketch guarantee: ε-approximate quantiles
  - If true rank is r, sketch gives r ± εn
  - Typically ε = 0.01-0.05 is sufficient
```

**Implementation**: XGBoost uses weighted GK sketch:

```cpp
// Each summary maintains sorted tuples (value, rank, delta)
// Merge with another summary:
void Merge(Summary const& b);

// Get approximate quantile:
float Query(float alpha);  // alpha ∈ [0, 1]
```

**booste-rs opportunity**: Implement streaming quantile sketch for memory efficiency.

### 3. Pre-sorted Feature Index

**The Problem**: Row-wise histogram building has poor cache locality.

**The Solution**: Pre-compute sorted feature indices.

```cpp
// GHistIndexMatrix stores:
// - index[row][col] = bin_id for feature col at row
// - row_ptr[row] = start offset for row in CSR format
```

**Cache-friendly access**: Process one feature at a time across all rows.

### 4. SIMD Histogram Building

**The Problem**: Individual gradient accumulation is slow.

**The Solution**: Use SIMD for parallel accumulation.

```cpp
// Process 8 bin updates at once with AVX2
// Each gradient pair is 8 bytes (2×float)
// AVX2 can process 256 bits = 32 bytes = 4 gradient pairs
```

**booste-rs opportunity**: Use packed_simd2 or std::simd for vectorization.

### 5. Parallel Histogram Building

**The Problem**: Single-threaded histogram building bottlenecks training.

**The Solution**: Build histograms in parallel, then merge.

```cpp
// Strategy 1: Partition rows across threads
// Each thread builds partial histogram
// Merge: hist[bin] = Σ thread_hist[t][bin]

// Strategy 2: Partition features across threads
// Each thread builds histogram for subset of features
// No merge needed (independent bins)
```

**booste-rs opportunity**: Rayon for parallel iteration, atomic accumulators.

### 6. Block-based Row Partitioning

**The Problem**: Partitioning rows after split has poor memory access.

**The Solution**: Use blocks for cache-friendly partitioning.

```cpp
constexpr size_t kBlockSize = 2048;

// Phase 1: Scan blocks, count left/right per block
for (block in blocks) {
  left_count[block] = count(rows[block], go_left);
}

// Phase 2: Compute block positions
for (block in blocks) {
  left_pos[block] = prefix_sum(left_count, block);
  right_pos[block] = offset + prefix_sum(right_count, block);
}

// Phase 3: Write rows to new positions
for (block in blocks) {
  for (row in rows[block]) {
    if (go_left(row)) {
      out[left_pos[block]++] = row;
    } else {
      out[right_pos[block]++] = row;
    }
  }
}
```

**booste-rs opportunity**: Block-based parallel partition with Rayon.

### 7. Missing Value Handling

**The Problem**: Real data has missing values.

**The Solution**: Learn optimal direction for missing values.

```cpp
// During split evaluation, try both directions:
// 1. Missing goes left: G_L = G_present_left + G_missing
// 2. Missing goes right: G_R = G_present_right + G_missing
// Choose direction with higher gain
```

**booste-rs opportunity**: Store `default_left` per split.

### 8. Sparse-Aware Processing

**The Problem**: Many datasets are sparse. Processing zeros is wasteful.

**The Solution**: Skip zero/missing values in gradient accumulation.

```cpp
// CSR format: only store non-zero entries
// For histogram: only accumulate present values
for (nz_idx = row_ptr[row]; nz_idx < row_ptr[row+1]; nz_idx++) {
  int col = col_idx[nz_idx];
  int bin = bin_idx[nz_idx];
  hist[col * n_bins + bin] += grad[row];
}
```

**booste-rs opportunity**: Sparse gradient computation.

## Memory Optimizations

### 1. Histogram Caching

**Problem**: Building histograms is expensive.

**Solution**: Cache and reuse histograms.

```cpp
class HistogramCache {
  // Map from node_id to histogram
  std::unordered_map<int, std::vector<GradientPair>> cache_;
  
  // Ring buffer for depth-wise: only need current + parent level
  std::array<Level, 2> levels_;
};
```

**booste-rs opportunity**: Simple LRU or ring buffer cache.

### 2. Compact Gradient Storage

**Problem**: Full precision gradients (2×f64) use lots of memory.

**Solution**: Use f32 for gradient/hessian pairs.

```cpp
struct GradientPair {
  float grad;  // Not double
  float hess;
};
// 8 bytes per sample instead of 16
```

**booste-rs opportunity**: Use f32 by default, f64 optional.

### 3. Compressed Bin Index

**Problem**: Bin indices could be u32 but often < 256 bins.

**Solution**: Use u8 when possible.

```cpp
// If max_bins <= 256, use uint8_t
// If max_bins <= 65536, use uint16_t
// Otherwise, use uint32_t
```

**booste-rs opportunity**: Enum over index types with const generics.

### 4. External Memory

**Problem**: Dataset doesn't fit in RAM.

**Solution**: Stream data from disk.

```text
XGBoost supports external memory DMatrix:
1. Quantize in batches
2. Stream batches for histogram building
3. Memory-map when possible
```

**booste-rs opportunity**: memory-mapped files with memmap2.

## Implementation Optimizations

### 1. Branch-free Split Check

**Problem**: Conditional branches are slow.

**Solution**: Use arithmetic instead of branches.

```cpp
// Branchy:
if (fvalue < split_value || is_missing && default_left) {
  go_left = true;
}

// Branchless:
int not_missing = !is_nan(fvalue);
int less_than = fvalue < split_value;
int go_left = (not_missing & less_than) | (!not_missing & default_left);
```

### 2. Prefetching

**Problem**: Cache misses stall the CPU.

**Solution**: Prefetch data before it's needed.

```cpp
// In histogram building loop:
for (size_t i = 0; i < n; i++) {
  _mm_prefetch(&hist[bin_idx[i + 16]], _MM_HINT_T0);
  hist[bin_idx[i]] += grad[i];
}
```

### 3. Loop Unrolling

**Problem**: Loop overhead is significant for tight loops.

**Solution**: Unroll loops manually or let compiler unroll.

```cpp
// Process 4 rows at a time
for (size_t i = 0; i < n; i += 4) {
  hist[bin[i+0]] += grad[i+0];
  hist[bin[i+1]] += grad[i+1];
  hist[bin[i+2]] += grad[i+2];
  hist[bin[i+3]] += grad[i+3];
}
```

### 4. Thread Pool

**Problem**: Creating threads has overhead.

**Solution**: Use thread pool for all parallel work.

```cpp
// XGBoost uses OpenMP thread pool
#pragma omp parallel for schedule(static)
for (int i = 0; i < n; i++) {
  // work
}
```

**booste-rs opportunity**: Rayon provides this automatically.

## Summary: Key Optimizations for booste-rs

### Must Have (High Impact)

| Optimization | Impact | Difficulty |
|--------------|--------|------------|
| Histogram subtraction | ~30% speedup | Easy |
| Parallel histogram building | ~Nx speedup | Medium |
| Block-based partitioning | ~2x speedup | Medium |
| f32 gradient pairs | 2x memory | Easy |
| u8/u16 bin indices | 2-4x memory | Easy |

### Should Have (Medium Impact)

| Optimization | Impact | Difficulty |
|--------------|--------|------------|
| Quantile sketching | Memory efficiency | Medium |
| Histogram caching | Avoid rebuild | Easy |
| Missing value direction | Accuracy | Easy |

### Nice to Have (Lower Impact)

| Optimization | Impact | Difficulty |
|--------------|--------|------------|
| SIMD histogram | ~2x inner loop | Hard |
| External memory | Large datasets | Hard |
| Prefetching | Cache efficiency | Medium |

## Source Code References

| Optimization | XGBoost Source |
|--------------|----------------|
| Histogram subtraction | `src/tree/updater_quantile_hist.cc:SubtractionTrick` |
| Quantile sketch | `src/common/quantile.h` |
| SIMD histogram | `src/common/hist_util.cc` |
| Block partitioning | `src/tree/common_row_partitioner.h` |
| Histogram cache | `src/tree/hist/histogram.h` |
