# RFC Index

Design documents for boosters, ordered by learning path.

## Implemented

| RFC | Topic | Description |
|-----|-------|-------------|
| [0001](./0001-data-matrix.md) | Data Matrix | Input data abstraction and layouts |
| [0002](./0002-forest-and-tree-structures.md) | Forest & Trees | Ensemble and tree data structures |
| [0003](./0003-inference-pipeline.md) | Inference | Prediction pipeline and traversal |
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

## Archive

Previous RFC versions are preserved in [archive/](./archive/) for reference.

## Creating New RFCs

Use [TEMPLATE.md](./TEMPLATE.md) as a starting point.
