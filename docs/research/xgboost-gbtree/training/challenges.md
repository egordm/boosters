# Implementation Challenges for booste-rs GBTree Training

## Overview

This document outlines the key challenges we'll face implementing GBTree training
in Rust and potential approaches to address them.

## Challenge 1: Quantization Pipeline

### The Challenge

XGBoost's quantization is tightly integrated with DMatrix construction. We need
to decide when and how to quantize.

```text
XGBoost flow:
  raw data → DMatrix → QuantileDMatrix (lazy, on first use)
  
booste-rs options:
  1. Quantize on construction (eager)
  2. Quantize on first train call (lazy)
  3. Separate QuantizedDataset type
```

### Design Decisions Needed

1. **Quantization timing**: Eager vs lazy vs explicit
2. **Bin count**: Fixed (256) or configurable
3. **Sketch algorithm**: Simple sort vs GK sketch vs t-digest
4. **Memory layout**: CSR vs dense vs hybrid

### Proposed Approach

- Start with **eager quantization** on explicit call
- Use **simple sorting** for initial implementation (exact quantiles)
- Support **configurable bin count** (default 256)
- Use **dense layout** initially (simpler, good for moderate features)

```rust
// Proposed API
let quantized = QuantizedMatrix::from_dense(
    &features,      // &[f32] or &Array2<f32>
    QuantizeConfig {
        max_bins: 256,
        missing: f32::NAN,
    },
)?;

// Later: streaming sketch for large datasets
let quantized = QuantizedMatrix::from_iter_sketch(
    batches,
    QuantizeConfig { max_bins: 256, .. },
)?;
```

## Challenge 2: Histogram Data Structure

### The Challenge

Histograms must be efficient to build, access, and subtract. The layout impacts
both cache efficiency and SIMD potential.

```text
Layout options:
  1. [feature][bin] - Feature-major
  2. [bin][feature] - Bin-major  
  3. [node][feature][bin] - Node-major
```

### Trade-offs

| Layout | Build | Subtract | Cache | SIMD |
|--------|-------|----------|-------|------|
| Feature-major | Random | Sequential | Good for split eval | Hard |
| Bin-major | Sequential | Random | Good for histogram build | Good |
| Node-major | Depends | Easy | Memory locality | Depends |

### Proposed Approach

- Use **feature-major** layout: `[feature * n_bins + bin]`
- Store `GradientPair` (g, h) at each position
- Use contiguous `Vec<GradientPair>` for cache efficiency

```rust
struct Histogram {
    // [feature][bin] layout, row-major
    data: Vec<GradientPair>,
    n_features: usize,
    n_bins: usize,
}

impl Histogram {
    fn get(&self, feature: usize, bin: usize) -> GradientPair {
        self.data[feature * self.n_bins + bin]
    }
    
    fn accumulate(&mut self, feature: usize, bin: usize, grad: GradientPair) {
        self.data[feature * self.n_bins + bin] += grad;
    }
    
    fn subtract_from(&self, parent: &Histogram) -> Histogram {
        // Element-wise subtraction
    }
}
```

## Challenge 3: Parallel Histogram Building

### The Challenge

Histogram building is the bottleneck. We need parallel builds without data races.

```text
Approaches:
  1. Thread-local histograms + merge
  2. Atomic accumulation
  3. Partition features across threads
  4. Partition rows across threads
```

### Trade-offs

| Approach | Contention | Memory | Merge Cost |
|----------|------------|--------|------------|
| Thread-local + merge | None | O(threads × bins) | O(threads × bins) |
| Atomics | High | O(bins) | None |
| Feature partition | None | O(bins) | None |
| Row partition + merge | None | O(threads × bins) | O(threads × bins) |

### Proposed Approach

- Use **row partition** with thread-local histograms for initial implementation
- Rayon's `par_chunks` for row partitioning
- Reduce thread-local histograms at the end

```rust
fn build_histogram_parallel(
    rows: &[RowIdx],
    gradients: &[GradientPair],
    gmat: &QuantizedMatrix,
) -> Histogram {
    rows.par_chunks(BLOCK_SIZE)
        .map(|chunk| {
            let mut local_hist = Histogram::zeros(n_features, n_bins);
            for &row in chunk {
                for (feature, bin) in gmat.row_bins(row) {
                    local_hist.accumulate(feature, bin, gradients[row]);
                }
            }
            local_hist
        })
        .reduce(Histogram::zeros, |a, b| a.add(&b))
}
```

## Challenge 4: Row Partitioning

### The Challenge

After a split, we need to partition rows into left and right children efficiently.

```text
Requirements:
  - Update row sets for all expanded nodes
  - Maintain row indices for histogram building
  - Handle missing values (default direction)
```

### Design Decisions Needed

1. **Storage**: Vec per node vs flat array with pointers
2. **In-place vs copy**: Swap rows or allocate new array
3. **Block size**: For cache efficiency

### Proposed Approach

- Use **flat array with ranges** for memory efficiency
- **In-place partitioning** with auxiliary buffer
- Block-based parallel partition

```rust
struct RowPartition {
    // Flat storage of row indices
    indices: Vec<RowIdx>,
    // [node_id] -> (start, end) range in indices
    node_ranges: Vec<(usize, usize)>,
}

impl RowPartition {
    fn partition_node(
        &mut self,
        node_id: NodeId,
        left_child: NodeId,
        right_child: NodeId,
        go_left: impl Fn(RowIdx) -> bool,
    ) {
        let (start, end) = self.node_ranges[node_id];
        let rows = &mut self.indices[start..end];
        
        // Partition in place
        let split_point = partition_stable(rows, |&r| go_left(r));
        
        self.node_ranges[left_child] = (start, start + split_point);
        self.node_ranges[right_child] = (start + split_point, end);
    }
}
```

## Challenge 5: Split Finding

### The Challenge

For each node and feature, we need to find the best split point by enumerating
all possible bin boundaries.

```text
Complexity: O(nodes × features × bins) per level
  - With histogram subtraction: ~1.5 × features × bins per node
  - Parallelizable across features
```

### Design Decisions Needed

1. **Missing value handling**: Enumerate both directions or use heuristic
2. **Split struct**: What to store (feature, threshold, gain, default_left)
3. **Regularization**: L1/L2 in gain formula

### Proposed Approach

- **Full enumeration** of missing directions (try both)
- Store complete split information for tree construction
- Support configurable regularization

```rust
struct SplitCandidate {
    feature: FeatureIdx,
    bin_idx: BinIdx,
    threshold: f32,        // Actual threshold value
    gain: f32,
    default_left: bool,
    left_sum: GradientPair,
    right_sum: GradientPair,
}

fn find_best_split(
    hist: &Histogram,
    node_sum: GradientPair,
    params: &TrainParams,
) -> Option<SplitCandidate> {
    (0..n_features)
        .into_par_iter()
        .filter_map(|feature| {
            find_best_split_for_feature(hist, feature, node_sum, params)
        })
        .max_by(|a, b| a.gain.partial_cmp(&b.gain).unwrap())
}
```

## Challenge 6: Tree Construction

### The Challenge

We need to build an efficient tree structure during training that can be
converted to our inference-optimized SoA format.

```text
Training needs:
  - Add nodes dynamically
  - Track parent-child relationships
  - Store split info + leaf values

Inference needs:
  - Compact SoA layout
  - Fast traversal
```

### Proposed Approach

- Use **mutable tree** during training with AoS layout
- Convert to **SoA format** after training complete

```rust
// Training tree (mutable, AoS)
struct TrainingTree {
    nodes: Vec<TrainingNode>,
}

enum TrainingNode {
    Leaf {
        value: f32,
        sum_gradients: GradientPair,
        row_count: usize,
    },
    Split {
        feature: FeatureIdx,
        threshold: f32,
        default_left: bool,
        left_child: NodeIdx,
        right_child: NodeIdx,
        gain: f32,
    },
}

impl TrainingTree {
    fn to_inference_tree(self) -> SoATreeStorage<ScalarLeaf> {
        // Convert to SoA layout for efficient inference
    }
}
```

## Challenge 7: Memory Management

### The Challenge

Training can use significant memory:

- Quantized matrix: O(n × d) bytes (1 byte per cell with u8 bins)
- Gradients: O(n × 8) bytes (GradientPair per sample)
- Histograms: O(nodes × d × bins × 8) bytes
- Row indices: O(n × 4) bytes (u32 per sample)

For 1M samples, 100 features, 256 bins:
- Quantized: 100 MB
- Gradients: 8 MB
- Histograms (per level): 200 KB × nodes
- Row indices: 4 MB

### Proposed Approach

- **Reuse histogram memory** across levels (ring buffer)
- **Stream gradients** if needed (recompute vs store)
- **Configurable precision** (f32 vs f64 gradients)

```rust
struct HistogramPool {
    // Ring buffer: only keep current + parent level
    levels: [LevelHistograms; 2],
    current_level: usize,
}

impl HistogramPool {
    fn get_current_level(&mut self) -> &mut LevelHistograms {
        &mut self.levels[self.current_level % 2]
    }
    
    fn advance_level(&mut self) {
        self.current_level += 1;
        // Current level's histograms become "parent" level
        // Old parent level can be reused
    }
}
```

## Challenge 8: API Design

API design will be addressed in the RFC phase. Key considerations:

- Easy to use for common cases
- Flexible for advanced use cases
- Compatible with our inference API
- Follows Rust idioms

## Priority Order

Based on dependencies and impact:

1. **Quantization** - Foundation for everything
2. **Histogram data structure** - Core abstraction
3. **Split finding** - Core algorithm
4. **Tree construction** - Produces output
5. **Row partitioning** - Required for multi-level trees
6. **Parallel histogram** - Performance optimization
7. **Memory management** - Scalability
8. **API design** - User-facing, can evolve

## Summary

Key design decisions to make before implementation:

| Decision | Recommendation | Rationale |
|----------|----------------|-----------|
| Quantization timing | Explicit call | Clear, predictable |
| Histogram layout | Feature-major | Good for split finding |
| Parallel strategy | Row partition + reduce | Rayon-friendly |
| Row partition storage | Flat array + ranges | Memory efficient |
| Training tree format | AoS (mutable) | Easier to build |
| Gradient precision | f32 default | Memory efficient |

These decisions balance implementation simplicity with performance. We can
revisit and optimize as we gain experience with real workloads.
