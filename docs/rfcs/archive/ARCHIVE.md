# RFC Archive

This folder contains the historical RFC set used during early development.

- Some RFCs are **superseded** or **refactor-only**.
- Some RFCs describe APIs that no longer exist.

The current RFC set lives in `docs/rfcs/` and is written with hindsight.

## Mapping (old → current)

### Data Layer

| Historical RFC | Current RFC | Notes |
| --- | --- | --- |
| RFC-0001 (Data Types and Layout) | [0001](../0001-dataset.md) Dataset | Core data abstractions |
| RFC-0019 (Feature Value Iterator) | [0001](../0001-dataset.md) Dataset | `for_each_feature_value` patterns |
| RFC-0021 (Dataset Separation) | [0001](../0001-dataset.md) Dataset | Dataset/BinnedDataset split rationale |
| RFC-0004 (Binning and Histograms) | [0002](../0002-binning.md) Binning | `BinMapper`, histogram building |
| RFC-0018 (Raw Feature Storage) | [0002](../0002-binning.md) Binning | Storage type selection |
| RFC-0011 (Feature Bundling) | [0003](../0003-efb.md) EFB | Bundling algorithm |
| RFC-0017 (EFB Training Integration) | [0003](../0003-efb.md) EFB | Effective views |
| RFC-0020 (EFB Architecture Cleanup) | [0003](../0003-efb.md) EFB | Architecture simplification |

### Trees and Training

| Historical RFC | Current RFC | Notes |
| --- | --- | --- |
| RFC-0002 (Forest and Tree Structures) | [0004](../0004-trees.md) Trees | SoA layout, `TreeView` |
| RFC-0005 (Tree Growing) | [0005](../0005-gbdt-training.md) GBDT Training | Split finding, tree construction |
| RFC-0006 (Training Configuration) | [0005](../0005-gbdt-training.md) + [0014](../0014-objectives-metrics.md) | Objectives, metrics, config |
| RFC-0016 (Single-Threaded Optimization) | [0013](../0013-histograms.md) Histograms | Histogram optimization |

### Inference

| Historical RFC | Current RFC | Notes |
| --- | --- | --- |
| RFC-0003 (Inference Pipeline) | [0006](../0006-gbdt-inference.md) GBDT Inference | `Predictor`, block processing |

### Linear Models

| Historical RFC | Current RFC | Notes |
| --- | --- | --- |
| RFC-0008 (GBLinear) / RFC-0009 | [0007](../0007-gblinear.md) GBLinear | Linear booster |
| RFC-0010 (Linear Trees) | [0008](../0008-linear-leaves.md) Linear Leaves | WLS at leaves |

### Advanced Features

| Historical RFC | Current RFC | Notes |
| --- | --- | --- |
| RFC-0012 (Native Categorical) | [0009](../0009-categoricals.md) Categoricals | Category splits |
| RFC-0013 (Explainability) | [0010](../0010-explainability.md) Explainability | Importance, SHAP |

### Integration

| Historical RFC | Current RFC | Notes |
| --- | --- | --- |
| RFC-0007 (Model Compatibility) | Removed | Will be deprecated, not migrated |
| RFC-0014 (Python Bindings) | [0011](../0011-python-bindings.md) Python | PyO3 bindings |
| RFC-0015 (Evaluation Framework) | [0012](../0012-evaluation-framework.md) Evaluation | boosters-eval |

## New RFCs (No Historical Equivalent)

These RFCs document designs that were implicit or scattered:

| Current RFC | Description |
| --- | --- |
| [0013](../0013-histograms.md) | Histogram building optimization (from RFC-0016) |
| [0014](../0014-objectives-metrics.md) | Objectives and Metrics (extracted from RFC-0006) |
| [0015](../0015-sampling.md) | Sampling strategies (extracted from RFC-0006) |

## Notes

- The historical RFC files keep their original filenames and content.
- New work should reference only the current RFC set.
- The new RFCs consolidate related topics and are written with implementation hindsight.
- Model Compatibility RFC was removed per user request (will be deprecated).

