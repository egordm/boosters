# booste-rs RFCs

This directory contains design documents (RFCs) for the booste-rs library.

**Last updated**: 2025-01-25

## RFC Index

### Core (Data & Trees)

| RFC | Title | Status | Description |
|-----|-------|--------|-------------|
| [0001](0001-data-types.md) | Data Types | Implemented | Layout convention, view types, accessor traits, Dataset, BinnedData |
| [0002](0002-forest-and-tree-structures.md) | Forest and Trees | Implemented | SoA tree representation, TreeView trait, MutableTree |
| [0003](0003-inference-pipeline.md) | Inference Pipeline | Implemented | Batch/single-row prediction, traversal strategies |

### Training (GBDT)

| RFC | Title | Status | Description |
|-----|-------|--------|-------------|
| [0004](0004-binning-and-histograms.md) | Binning and Histograms | Implemented | Quantization, BinnedDataset, histogram accumulation |
| [0005](0005-tree-growing.md) | Tree Growing | Implemented | Split finding, tree construction, depth/leaf-wise growth |
| [0006](0006-training-configuration.md) | Training Configuration | Implemented | Objectives, metrics, sampling, multi-output |

### Extensions

| RFC | Title | Status | Description |
|-----|-------|--------|-------------|
| [0007](0007-model-compatibility.md) | Model Compatibility | Implemented | XGBoost/LightGBM model loading |
| [0008](0008-arrow-parquet-io.md) | Arrow/Parquet I/O | Implemented | Data loading from Arrow IPC and Parquet |
| [0009](0009-gblinear.md) | GBLinear | Implemented | Linear booster with coordinate descent |
| [0010](0010-linear-trees.md) | Linear Leaves | Implemented | Linear model fitting at tree leaves |
| [0011](0011-feature-bundling.md) | Feature Bundling | Implemented | EFB for sparse/one-hot features |
| [0012](0012-native-categorical-features.md) | Native Categorical | Implemented | Native categorical feature handling |
| [0013](0013-explainability.md) | Explainability | Draft | Feature importance and SHAP values |

## Reading Order

For understanding the system architecture:

1. **Data Layer**: RFC-0001 (Data Types)
2. **Model Layer**: RFC-0002 (Trees), RFC-0009 (GBLinear)
3. **Inference**: RFC-0003 (Pipeline)
4. **Training**: RFC-0004 (Binning) → RFC-0005 (Growing) → RFC-0006 (Config)
5. **Advanced**: RFC-0010 (Linear Leaves), RFC-0011 (Bundling), RFC-0012 (Categoricals)

## RFC Status

- **Draft**: Design in progress, open for feedback
- **Accepted**: Design approved, implementation pending
- **Implemented**: Code complete and tested
- **Deprecated**: Superseded by newer RFC

## Archive

Previous RFC versions before the 2025-01-24 restructure are in [archive/pre-restructure/](archive/pre-restructure/).

## Creating New RFCs

Use [TEMPLATE.md](TEMPLATE.md) as a starting point.
