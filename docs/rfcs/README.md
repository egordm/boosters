# booste-rs RFCs

Design documents for booste-rs. Each RFC describes the design of a major component.

Historical RFCs (superseded or refactor-only documents) are in `docs/rfcs/archive/`.

**Last updated**: 2026-01-02

## RFC Index

| RFC | Title | Status | Description |
| --- | ----- | ------ | ----------- |
| [0001](0001-dataset.md) | Dataset | Implemented | User-facing data container, feature-major layout, sparse support |
| [0002](0002-trees.md) | Trees and Forest | Implemented | SoA layout, `Tree<L>`, `MutableTree`, multi-output groups |
| [0003](0003-binning.md) | Binning | Implemented | Quantization for histograms, `BinnedDataset`, bin types |
| [0004](0004-efb.md) | Exclusive Feature Bundling | Implemented | Sparse feature bundling, conflict detection, offset encoding |
| [0005](0005-objectives-metrics.md) | Objectives and Metrics | Implemented | `ObjectiveFn`, `MetricFn`, early stopping |
| [0006](0006-sampling.md) | Sampling Strategies | Implemented | Row (uniform, GOSS) and column sampling |
| [0007](0007-histograms.md) | Histogram Building | Implemented | Histogram optimization, subtraction trick, LRU cache |
| [0008](0008-gbdt-training.md) | GBDT Training | Implemented | `GBDTTrainer`, grower, split finding, growth strategies |
| [0009](0009-gbdt-inference.md) | GBDT Inference | Implemented | `Predictor`, block processing, `TreeTraversal` trait |
| [0010](0010-gblinear.md) | GBLinear | Implemented | `LinearModel`, coordinate descent, feature selection |
| [0011](0011-linear-leaves.md) | Linear Leaves | Implemented | Linear models at tree leaves, WLS fitting |
| [0012](0012-categoricals.md) | Categorical Features | Implemented | One-hot, sorted partition, `CatBitset` |
| [0013](0013-explainability.md) | Explainability | Implemented | Feature importance, TreeSHAP, Linear SHAP |
| [0014](0014-python-bindings.md) | Python Bindings | Implemented | PyO3 bindings, sklearn estimators |
| [0015](0015-evaluation-framework.md) | Evaluation Framework | Implemented | `boosters-eval` quality benchmarks |

## Reading Order

1. **Data**: [0001](0001-dataset.md) Dataset → [0002](0002-trees.md) Trees → [0003](0003-binning.md) Binning → [0004](0004-efb.md) EFB
2. **Training**: [0005](0005-objectives-metrics.md) Objectives → [0006](0006-sampling.md) Sampling → [0007](0007-histograms.md) Histograms → [0008](0008-gbdt-training.md) Training
3. **Inference**: [0009](0009-gbdt-inference.md) Inference
4. **Linear**: [0010](0010-gblinear.md) GBLinear → [0011](0011-linear-leaves.md) Linear Leaves
5. **Advanced**: [0012](0012-categoricals.md) Categoricals → [0013](0013-explainability.md) Explainability
6. **Python**: [0014](0014-python-bindings.md) Bindings → [0015](0015-evaluation-framework.md) Eval Framework

## Layers

| Layer | Description | User |
| ----- | ----------- | ---- |
| **High** | `GBDTModel`, `GBLinearModel` | End users |
| **Medium** | Trainer, Predictor, Sampler | Power users |
| **Low** | Histogram ops, split finding | Optimization |

## Status Legend

- **Implemented**: In `crates/boosters` and/or `packages/*`
- **Draft**: Design in progress
- **Planned**: Design sketched, not started

## Archive

See `docs/rfcs/archive/ARCHIVE.md` for archived RFCs.
