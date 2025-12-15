# RFC Index

Request for Comments (RFCs) document the design decisions for booste-rs.

## Implemented

| RFC | Topic | Description |
|-----|-------|-------------|
| [0001](./0001-forest-data-structures.md) | Forest Structures | Multi-tree ensemble representation |
| [0002](./0002-tree-data-structures.md) | Tree Structures | Individual decision tree layout |
| [0003](./0003-visitor-and-traversal.md) | Traversal | Tree traversal and prediction |
| [0004](./0004-dmatrix.md) | DMatrix | Data input abstraction |
| [0007](./0007-serialization.md) | Serialization | XGBoost JSON model loading |
| [0008](./0008-gblinear-inference.md) | Linear Inference | GBLinear prediction |
| [0009](./0009-gblinear-training.md) | Linear Training | GBLinear coordinate descent |
| [0010](./0010-matrix-layouts.md) | Matrix Layouts | Row/column-major storage |
| [0011](./0011-quantization-binning.md) | Quantization | Feature binning for histograms |
| [0012](./0012-histogram-building.md) | Histograms | Gradient histogram construction |
| [0013](./0013-split-finding.md) | Split Finding | Best split selection |
| [0014](./0014-row-partitioning.md) | Row Partitioning | Sample assignment to nodes |
| [0015](./0015-tree-growing.md) | Tree Growing | Level-wise and leaf-wise strategies |
| [0016](./0016-categorical-training.md) | Categorical | Native categorical features |
| [0017](./0017-sampling-strategies.md) | Sampling | GOSS, row/column sampling |
| [0018](./0018-multi-output-trees.md) | Multi-Output Trees | Vector leaf support |
| [0019](./0019-feature-bundling.md) | Feature Bundling | Exclusive feature bundling |
| [0020](./0020-evaluation-metrics.md) | Evaluation | Training metrics |
| [0021](./0021-base-score-init.md) | Base Score | Initial prediction setup |
| [0022](./0022-unified-multi-output.md) | Multi-Output | One-tree-per-output strategy |
| [0026](./0026-sample-weighting.md) | Sample Weights | Weighted training |
| [0028](./0028-prediction-outputs-and-transforms.md) | Prediction Output | Raw/transformed predictions |

## Draft / Planned

| RFC | Topic | Status |
|-----|-------|--------|
| [0023](./0023-constraints.md) | Constraints | Delayed |
| [0025](./0025-row-parallel-histograms.md) | Row-Parallel | Draft |
| [0027](./0027-gradient-quantization.md) | Gradient Quantization | Draft |
| [0029](./0029-arrow-datasets.md) | Arrow I/O | Draft |

## Creating New RFCs

Use [TEMPLATE.md](./TEMPLATE.md) as a starting point.
