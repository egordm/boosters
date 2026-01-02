# booste-rs RFCs

This directory contains design documents (RFCs) for the booste-rs library.

**Last updated**: 2026-01-02

## RFC Index

### Core (Data & Trees)

| RFC | Title | Status | Description |
| --- | ----- | ------ | ----------- |
| [0001](0001-data-types.md) | Data Types | Implemented | Layout convention, view types, accessor traits, Dataset, BinnedData |
| [0002](0002-forest-and-tree-structures.md) | Forest and Trees | Implemented | SoA tree representation, TreeView trait, MutableTree |
| [0003](0003-inference-pipeline.md) | Inference Pipeline | Implemented | Batch/single-row prediction, traversal strategies |

### Training (GBDT)

| RFC | Title | Status | Description |
| --- | ----- | ------ | ----------- |
| [0004](0004-binning-and-histograms.md) | Binning and Histograms | Implemented | Quantization, BinnedDataset, histogram accumulation |
| [0005](0005-tree-growing.md) | Tree Growing | Implemented | Split finding, tree construction, depth/leaf-wise growth |
| [0006](0006-training-configuration.md) | Training Configuration | Implemented | Objectives, metrics, sampling, multi-output |

### Extensions

| RFC | Title | Status | Description |
| --- | ----- | ------ | ----------- |
| [0007](0007-model-compatibility.md) | Model Compatibility | Implemented | XGBoost/LightGBM model loading |
| [0008](0008-arrow-parquet-io.md) | Arrow/Parquet I/O | Superseded | Historical Arrow/Parquet I/O design; superseded by RFC-0021 |
| [0009](0009-gblinear.md) | GBLinear | Implemented | Linear booster with coordinate descent |
| [0010](0010-linear-trees.md) | Linear Leaves | Implemented | Linear model fitting at tree leaves |
| [0011](0011-feature-bundling.md) | Feature Bundling | Implemented | EFB for sparse/one-hot features |
| [0012](0012-native-categorical-features.md) | Native Categorical | Implemented | Native categorical feature handling |
| [0013](0013-explainability.md) | Explainability | Draft | Feature importance and SHAP values |
| [0014](0014-python-bindings.md) | Python Bindings | Implemented | PyO3 bindings + sklearn-style API |
| [0015](0015-evaluation-framework.md) | Evaluation Framework | Implemented | `boosters-eval` quality + benchmarking |
| [0016](0016-single-threaded-optimization.md) | Single-Threaded Optimization | Draft | Performance analysis + optimization plan |
| [0017](0017-efb-training-integration.md) | EFB Training Integration | Implemented | Integrate EFB into training pipeline |
| [0018](0018-raw-feature-storage.md) | Raw Feature Storage | Superseded | Historical design; superseded by RFC-0021 |
| [0019](0019-feature-value-iterator.md) | Feature Value Iterator | Implemented | Zero-cost feature-value iteration on `Dataset` |
| [0020](0020-efb-architecture-cleanup.md) | EFB Architecture Cleanup | Draft | Simplify EFB histogram building architecture |
| [0021](0021-dataset-separation.md) | Dataset Separation | Implemented | Split raw `Dataset` from internal `BinnedDataset` |

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

Superseded and historical RFCs are kept in-place (with status **Superseded**) so old discussions and benchmarks remain linkable.

## Creating New RFCs

Use [TEMPLATE.md](TEMPLATE.md) as a starting point.
