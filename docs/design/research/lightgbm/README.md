# LightGBM Research

Research on Microsoft's LightGBM implementation.

## Overview

LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework developed by
Microsoft that uses tree-based learning algorithms. It's known for its speed and efficiency,
particularly for large datasets.

## Key Differentiators from XGBoost

1. **Leaf-wise (best-first) tree growth** — Prioritizes highest-gain splits
2. **GOSS (Gradient-based One-Side Sampling)** — Sample based on gradient magnitude
3. **Native categorical feature handling** — O(k log k) optimal splits
4. **Feature bundling** — Combine mutually exclusive sparse features
5. **CPU gradient quantization** — 16/32-bit packed histograms

## Contents

### Training

Core algorithms and data structures for histogram-based tree building:

| Document | Description |
|----------|-------------|
| [Training Overview](training/overview.md) | High-level training pipeline |
| [Leaf-wise Growth](training/leaf_wise_growth.md) | Best-first tree growth strategy |
| [GOSS Sampling](training/goss.md) | Gradient-based one-side sampling |
| [Histogram Building](training/histogram_building.md) | Gradient histogram construction |
| [Categorical Features](training/categorical_features.md) | Native categorical handling |

### Data Structures

Key data structures (to be documented):

| Document | Description |
|----------|-------------|
| [BinMapper](data_structures/bin_mapper.md) | Feature binning and quantization |
| [Dataset](data_structures/dataset.md) | Training data storage |
| [Tree](data_structures/tree.md) | Tree structure |

### Inference

Prediction (to be documented):

| Document | Description |
|----------|-------------|
| [Prediction](inference/prediction.md) | Inference pipeline |

## Key Concepts

### Training Pipeline

```text
┌─────────────────────────────────────────────────────────────────────┐
│  Raw Features (f32)                                                 │
│       ↓                                                             │
│  BinMapper (greedy quantile binning)                                │
│       ↓                                                             │
│  Dataset (binned features, supports sparse + dense)                 │
│       ↓                                                             │
│  For each boosting round:                                           │
│    1. Compute gradients (from objective)                            │
│    2. [Optional] GOSS sampling (keep top gradients + random sample) │
│    3. Init root with all data                                       │
│    4. While (can split):                                            │
│       a. Find leaf with highest potential gain                      │
│       b. Build histogram for that leaf                              │
│       c. Find best split                                            │
│       d. Apply split, partition data                                │
│    5. Update predictions                                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Comparison with XGBoost

| Aspect | LightGBM | XGBoost |
|--------|----------|---------|
| Tree growth | Leaf-wise (best-first) | Depth-wise (default) |
| Sampling | GOSS (gradient-based) | Random subsampling |
| Categorical | Native (gradient-sorted) | Manual encoding |
| Quantization | CPU + GPU | GPU only |
| Histogram | Adaptive col/row | Row-wise default |
| Split gain | Same formula | Same formula |

### Source Code Map

| Component | Source Files |
|-----------|--------------|
| Main booster | `src/boosting/gbdt.{h,cpp}` |
| Tree learner | `src/treelearner/serial_tree_learner.{h,cpp}` |
| Histogram | `src/treelearner/feature_histogram.{hpp,cpp}` |
| Data partition | `src/treelearner/data_partition.hpp` |
| Split info | `src/treelearner/split_info.hpp` |
| GOSS | `src/boosting/goss.hpp` |
| Binning | `src/io/bin.cpp`, `include/LightGBM/bin.h` |
| Dataset | `src/io/dataset.cpp`, `include/LightGBM/dataset.h` |
| Tree | `src/io/tree.cpp`, `include/LightGBM/tree.h` |

## Related

See [XGBoost vs LightGBM Comparison](../comparison.md) for detailed comparison.
