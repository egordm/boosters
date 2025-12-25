# RFC Index

Design documents for boosters, ordered by learning path.

**Last updated**: 2025-01-23

## Implemented

| RFC | Topic | Description |
|-----|-------|-------------|
| [0001](./0001-data-matrix.md) | Data Matrix | ndarray-based feature views (SamplesView, FeaturesView) |
| [0002](./0002-forest-and-tree-structures.md) | Forest & Trees | SoA tree storage, TreeView trait, MutableTree |
| [0003](./0003-inference-pipeline.md) | Inference | Predictor with traversal strategies |
| [0004](./0004-quantization-and-binning.md) | Binning | Feature quantization for histograms |
| [0005](./0005-histogram-building.md) | Histograms | Gradient histogram construction |
| [0006](./0006-split-finding.md) | Split Finding | Best split selection with regularization |
| [0007](./0007-tree-growing.md) | Tree Growing | Depth-wise and leaf-wise strategies |
| [0008](./0008-objectives-and-losses.md) | Objectives | Loss functions and gradients |
| [0009](./0009-evaluation-metrics.md) | Metrics | Evaluation and early stopping |
| [0010](./0010-sampling-strategies.md) | Sampling | GOSS and column sampling |
| [0011](./0011-multi-output-training.md) | Multi-Output | Multiclass and multi-target |
| [0012](./0012-model-compatibility.md) | Compatibility | XGBoost and LightGBM loading |
| [0013](./0013-arrow-parquet-io.md) | Arrow/Parquet | Data loading from Arrow/Parquet |
| [0014](./0014-gblinear.md) | GBLinear | Linear booster training and inference |
| [0015](./0015-linear-trees.md) | Linear Leaves | Linear models at tree leaf nodes |
| [0017](./0017-feature-bundling.md) | Feature Bundling | Exclusive Feature Bundling (EFB) |
| [0018](./0018-native-categorical-features.md) | Categoricals | Native categorical feature support |

## Draft

| RFC | Topic | Description |
|-----|-------|-------------|
| [0019](./0019-dataset-format.md) | Dataset Format | High-level Dataset type |
| [0020](./0020-data-access-layer.md) | Data Access | View types and access patterns |
| [0022](./0022-explainability.md) | Explainability | Feature importance and SHAP values |

## Reading Order

For understanding the system architecture:

1. **Data Layer**: RFC-0001 (Data Matrix)
2. **Model Layer**: RFC-0002 (Trees), RFC-0014 (GBLinear)
3. **Inference**: RFC-0003 (Pipeline)
4. **Training**: RFC-0004→0007 (Binning→Histograms→Splits→Growing)
5. **Training Config**: RFC-0008 (Objectives), RFC-0009 (Metrics), RFC-0010 (Sampling)
6. **Advanced**: RFC-0015 (Linear Leaves), RFC-0017 (Bundling), RFC-0018 (Categoricals)

## Archived

Previous RFCs that have been superseded or absorbed into other documents:

| RFC | Reason |
|-----|--------|
| [0016](./archive/0016-prediction-architecture.md) | Content absorbed into RFC-0002 |

## Creating New RFCs

Use [TEMPLATE.md](./TEMPLATE.md) as a starting point.
