# GBTree Training Overview

## Introduction

XGBoost's histogram-based tree training (`hist` method) is the modern, efficient approach
for building gradient boosted trees. It replaces the older exact greedy method with an
approximate algorithm that bins continuous features into discrete histograms.

This document provides a high-level overview of the training pipeline and key components.

## Training Pipeline

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                         INITIALIZATION                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Quantile Sketch: Sample features → compute bin boundaries           │
│  2. Build HistogramCuts: Store cut points per feature                   │
│  3. Build GHistIndexMatrix: Quantize all features to bin indices        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      FOR EACH BOOSTING ROUND                            │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Compute Gradients: g_i = ∂L/∂ŷ_i, h_i = ∂²L/∂ŷ_i²                  │
│  2. Build Root Histogram: Aggregate (g, h) per bin for all rows         │
│  3. Find Best Root Split: Scan histogram bins, compute gain             │
│  4. Grow Tree:                                                          │
│     while (valid_candidates):                                           │
│       a. Apply Split: Expand node → create children                     │
│       b. Partition Rows: Assign each row to left or right child         │
│       c. Build Child Histograms:                                        │
│          - Build histogram for smaller child                            │
│          - Derive larger child via subtraction                          │
│       d. Evaluate Splits: Find best split for each child                │
│       e. Queue Valid: Add children with positive gain to queue          │
│  5. Update Predictions: Add tree's contribution to ŷ                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. QuantileDMatrix / HistogramCuts

Before training begins, XGBoost scans all features to determine bin boundaries using
quantile sketching. This ensures each bin contains approximately equal numbers of samples.

```text
Feature values:   [0.1, 0.5, 0.8, 1.2, 2.0, 5.0, 10.0]
After quantile binning (3 bins):
  Bin 0: [0.1, 0.8]    cut = 0.8
  Bin 1: [1.2, 2.0]    cut = 2.0
  Bin 2: [5.0, 10.0]   cut = ∞
```

See: [quantization.md](quantization.md)

### 2. GHistIndexMatrix

The quantized feature matrix stores bin indices (u8/u16/u32) instead of raw floats.
Supports both dense and sparse layouts.

```text
Original:  [[1.5, 3.2], [0.8, 2.1], [4.0, 1.0]]
Quantized: [[2, 3], [1, 2], [3, 1]]  (bin indices)
```

See: [../data_structures/gradient_index.md](../data_structures/gradient_index.md)

### 3. Histogram Building

For each node, aggregate gradients by bin index:

```text
hist[bin] = Σ (g_i, h_i) for all rows where feature falls in bin
```

The key insight: instead of sorting O(n) rows per split evaluation, we aggregate
into O(bins) histogram entries once, then scan bins in O(bins) time.

See: [histogram_building.md](histogram_building.md)

### 4. Split Finding

For each feature, scan the histogram bins and compute split gain:

```text
Gain = ½ [ G_L²/(H_L + λ) + G_R²/(H_R + λ) - (G_L+G_R)²/(H_L+H_R + λ) ] - γ

where G_L/G_R = sum of gradients in left/right child
      H_L/H_R = sum of hessians in left/right child
      λ = L2 regularization
      γ = min_split_loss (complexity cost)
```

See: [split_finding.md](split_finding.md), [gain_calculation.md](gain_calculation.md)

### 5. Row Partitioning

After a split is applied, rows must be assigned to left or right child. XGBoost
maintains a `RowSetCollection` that tracks which rows belong to each node.

See: [row_partitioning.md](row_partitioning.md)

### 6. Tree Growing Strategy

XGBoost supports two strategies:

- **Depth-wise** (default): Grow all nodes at the same depth before moving deeper
- **Loss-guided** (like LightGBM): Always expand the node with highest gain

See: [tree_growing.md](tree_growing.md)

## XGBoost Source Code Map

| Component | Source Files |
|-----------|--------------|
| Tree Updater (main entry) | `src/tree/updater_quantile_hist.cc` |
| Histogram Builder | `src/tree/hist/histogram.{h,cc}` |
| Split Evaluator | `src/tree/hist/evaluate_splits.h` |
| Row Partitioner | `src/tree/common_row_partitioner.h` |
| Tree Growing Driver | `src/tree/driver.h` |
| Gradient Index Matrix | `src/data/gradient_index.{h,cc}` |
| Histogram Cuts | `src/common/hist_util.{h,cc}` |
| Gain/Weight Calculation | `src/tree/param.h` |
| Training Parameters | `src/tree/param.h` |

## Complexity Analysis

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Quantization (one-time) | O(n × d × log(max_bin)) | n=rows, d=features |
| Histogram Build | O(n_node × d) | n_node = rows in node |
| Split Evaluation | O(d × bins) | Scan all bins per feature |
| Row Partition | O(n_node) | Assign each row once |
| Total per tree | O(n × d × depth) | Dominated by histogram building |

With histogram subtraction, only ~50% of nodes need explicit histogram building.

## Threading Model

XGBoost parallelizes at multiple levels:

1. **Across nodes**: At the same depth, multiple nodes can be processed in parallel
2. **Within histogram building**: Rows are processed in parallel with thread-local
   histograms, then merged
3. **Within split evaluation**: Features can be evaluated in parallel

```text
┌────────────────────────────────────────────────────────────────────┐
│  Node Level Parallelism (blocked space over nodes × rows)          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │  Node A          │  │  Node B          │  │  Node C          │ │
│  │  Thread 0-3      │  │  Thread 4-7      │  │  Thread 8-11     │ │
│  │  Local Histograms│  │  Local Histograms│  │  Local Histograms│ │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘ │
│           ↓                     ↓                     ↓           │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  Reduce: Merge thread-local histograms → final histogram    │   │
│  └────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

## Key Optimizations

1. **Histogram Subtraction**: child_hist = parent_hist - sibling_hist
2. **Prefetching**: Prefetch gradient pairs and bin indices during histogram building
3. **Dense vs Sparse Dispatch**: Different kernels for dense (no missing) vs sparse data
4. **Compressed Bin Indices**: Use u8/u16 when max_bin ≤ 256/65536
5. **Row-wise vs Column-wise Building**: Choose based on L2 cache fit
6. **Parallel Histogram Merge**: Thread-local histograms merged in parallel

## Next Steps

For detailed exploration of each component, see the linked documents above.
