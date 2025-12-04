# XGBoost GBTree Research

Research on XGBoost's tree-based booster implementation.

## Contents

### Training

Core algorithms and data structures for histogram-based tree building:

| Document | Description |
|----------|-------------|
| [Training Overview](training/overview.md) | High-level training pipeline and components |
| [Quantization & Sketching](training/quantization.md) | How continuous features become discrete bins |
| [Histogram Building](training/histogram_building.md) | Aggregating gradients per bin |
| [Split Finding](training/split_finding.md) | Evaluating and selecting best splits |
| [Row Partitioning](training/row_partitioning.md) | Assigning rows to tree nodes |
| [Tree Growing Strategies](training/tree_growing.md) | Depth-wise vs loss-guided growth |
| [Optimizations](training/optimizations.md) | Key XGBoost optimizations for training |
| [GPU Training](training/gpu_training.md) | CUDA-accelerated training |
| [Implementation Challenges](training/challenges.md) | Challenges and design decisions for booste-rs |

### Inference

Optimizations for fast prediction:

| Document | Description |
|----------|-------------|
| [Inference Pipeline](inference/xgboost-inference.md) | XGBoost prediction flow |
| [Array Tree Layout](inference/array_tree_layout.md) | SoA tree storage |
| [Block-based Traversal](inference/block_based_traversal.md) | Cache-efficient prediction |
| [Packed Vector Leaf](inference/packed_vector_leaf.md) | Multi-output leaf packing |
| [Precompute Pack Grouping](inference/precompute_pack_grouping.md) | Tree-to-group mapping |

### Data Structures

Shared data structures used in both training and inference:

| Document | Description |
|----------|-------------|
| [HistogramCuts](data_structures/histogram_cuts.md) | Bin boundaries for quantization |
| [Quantized Features](data_structures/quantized_features.md) | Feature quantization concepts |
| [GHistIndexMatrix](data_structures/gradient_index.md) | Quantized feature matrix |
| [QuantileDMatrix](data_structures/quantile_dmatrix.md) | Streaming quantization wrapper |
| [Storage Layouts](data_structures/storage_layouts.md) | Memory layouts for quantized data |

## Key Concepts

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  Raw Features (f32)                                                 │
│       ↓                                                             │
│  Quantile Sketch → HistogramCuts (bin boundaries per feature)       │
│       ↓                                                             │
│  GHistIndexMatrix (quantized bins, u8/u16/u32)                      │
│       ↓                                                             │
│  For each boosting round:                                           │
│    1. Compute gradients (from objective)                            │
│    2. Build histogram for root                                      │
│    3. Find best split for root                                      │
│    4. While candidates exist:                                       │
│       a. Apply splits → partition rows                              │
│       b. Build histograms for children                              │
│          (use subtraction trick: child = parent - sibling)          │
│       c. Find best splits for children                              │
│       d. Add valid candidates to queue                              │
│    5. Update predictions                                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Optimization Stack

**Training:**
1. **Quantization** — Convert continuous features to discrete bins (0-255)
2. **Histogram Building** — Aggregate gradients per bin instead of per row
3. **Histogram Subtraction** — Derive sibling histogram from parent - built
4. **Parallel Node Processing** — Process all nodes at same depth together

**Inference:**
1. **Array Tree Layout** — SoA storage for cache efficiency
2. **Block Processing** — Process rows in blocks (e.g., 64 rows)
3. **Unrolled Traversal** — Process all rows at same tree level together
4. **Thread Parallelism** — Parallelize across rows or blocks

## Lessons for booste-rs

Key takeaways from this research are documented in [Lessons Learned](lessons_learned.md):

- Algorithmic optimizations (histogram, subtraction) provide the biggest gains
- Separate training and inference tree representations
- Use f32 gradients and u8 bin indices by default
- Block-based parallelism with Rayon
- Builder pattern for configuration

See the full document for detailed recommendations.
