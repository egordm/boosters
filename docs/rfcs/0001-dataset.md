# RFC-0001: Dataset

**Status**: Implemented  
**Created**: 2025-12-15  
**Updated**: 2026-01-02  
**Scope**: User-facing raw data container

## Summary

`Dataset` is the primary user-facing data type. It stores:

- Raw feature values (dense or sparse, one column per feature)
- Optional targets (shape `[n_outputs, n_samples]`)
- Optional sample weights (length `n_samples`)
- Immutable schema / feature metadata

The core ideology (from RFC-0021): keep raw and binned data separate.

- `Dataset` is for raw-value access (prediction, SHAP, GBLinear, linear leaves, metrics).
- `BinnedDataset` is for training-only histogram building (bin indices + bin mappers).

## Goals

- Make the *common* access patterns fast and hard to misuse.
- Support sparse features without densifying the full matrix.
- Keep the API small and stable: add new access primitives only when they are clearly needed.

## Non-goals

- Store both feature-major and sample-major layouts permanently (avoids 2× memory).
- Provide a “universal iterator” abstraction over every access mode.

## Data Model

### Dataset storage (conceptual)

```rust
pub struct Dataset {
    features: Vec<Feature>,
    n_samples: usize,
    schema: DatasetSchema,
    targets: Option<Array2<f32>>, // [n_outputs, n_samples]
    weights: Option<Array1<f32>>, // [n_samples]
}

pub enum Feature {
    Dense(Array1<f32>),
    Sparse {
        indices: Vec<u32>,
        values: Vec<f32>,
        n_samples: usize,
        default: f32,
    },
}
```

### Sparse feature semantics and invariants

A sparse feature is “column-sparse”: it stores a sorted list of (row index, value) pairs.
All unspecified rows implicitly have value `default`.

Required invariants:

- `indices.len() == values.len()`
- `indices` are strictly increasing (sorted, no duplicates)
- each index is in bounds: `index < n_samples`

The builder validates these invariants and returns `DatasetError` on violations.

### Targets and weights

- Targets are optional; when present they are stored as `[n_outputs, n_samples]`.
- Weights are optional; when absent they are treated as uniform weight `1.0`.

## Why Feature-Major?

The library’s training loops are feature-dominant:

- GBDT histogram building iterates *one feature* across many samples.
- GBLinear / coordinate descent iterates *one feature* across many samples.

So we store features as columns (`[n_features, n_samples]` conceptually) to keep each feature contiguous.

Prediction and SHAP need sample-major access. Instead of storing both layouts, we use block transposition:

- `Dataset::buffer_samples()` fills a caller-owned buffer of shape `[block_size, n_features]`.
- The returned `SamplesView` is sample-major and cache-friendly for traversal.

## Access Patterns (the important part)

The key design choice is that we expose a few primitives that map directly to algorithmic needs.

### 1) Sparse-efficient iteration: `for_each_feature_value`

```rust
dataset.for_each_feature_value(feature_idx, |sample_idx, value| {
    // Dense: called n_samples times
    // Sparse: called nnz times (only stored non-default values)
});
```

Use when:

- You want to ignore implicit defaults (common for sparse-aware algorithms).
- You want “match once, loop tight” performance (no per-element branching).

Important semantics:

- For sparse features, this iterates *only* stored values.
- If you need a dense stream including defaults, use `for_each_feature_value_dense`.

### 2) Dense stream without materialization: `for_each_feature_value_dense`

```rust
dataset.for_each_feature_value_dense(feature_idx, |sample_idx, value| {
    // Always called exactly n_samples times.
});
```

Use when:

- Downstream code expects one value per sample (e.g., quantization/binning).
- You want to keep sparse storage but still “act dense” on demand.

### 3) Subset gather for leaves: `gather_feature_values`

```rust
dataset.gather_feature_values(feature_idx, sample_indices, buffer);
```

This is optimized for “give me feature values for a subset of rows”, and is used by linear leaves.

Critical requirement:

- `sample_indices` must be sorted ascending.
  - Dense features still work either way.
  - Sparse features use a merge-join; unsorted indices can produce incorrect results.

If you want to avoid writing into a buffer, use `for_each_gathered_value()`.

### 4) Sample-major blocks for traversal: `buffer_samples`

```rust
let mut block = ndarray::Array2::<f32>::zeros((block_size, dataset.n_features()));
let samples = dataset.buffer_samples(&mut block, start_sample);
// samples: SamplesView<'_> with shape [samples_filled, n_features]
```

Use when:

- You traverse trees row-by-row (GBDT prediction, TreeSHAP).
- You want to parallelize by sample blocks, with per-thread buffer reuse.

Semantics:

- The buffer must have shape `[block_size, n_features]`.
- The returned view covers only the filled prefix (end-of-dataset can be partial).

### Summary table

| Pattern | Method | Intended use |
| --- | --- | --- |
| Sparse-aware scan | `for_each_feature_value` | GBLinear, LinearSHAP, sparse feature stats |
| Dense stream without densifying | `for_each_feature_value_dense` | Binning/quantization |
| Subset gather | `gather_feature_values` / `for_each_gathered_value` | Linear leaves (per-leaf subsets) |
| Sample-major traversal | `buffer_samples` → `SamplesView` | Prediction, TreeSHAP |

## Missing Values

Missing values are represented as `f32::NAN`.

- Tree models handle missing via the split’s `default_left` direction.
- Sparse features can use `default = f32::NAN` to treat unspecified entries as missing.

## Categorical Features

Categorical features are stored as `f32`, but they are conceptually integer category IDs:

- Valid values are non-negative integers encoded as floats (e.g., `0.0, 1.0, 2.0`).
- The schema marks them as `FeatureType::Categorical`.
- Binning casts to integer internally for categorical split logic.

## Views

The data module provides lightweight read-only views with explicit semantics:

- `TargetsView`: target values with shape `[n_outputs, n_samples]`.
- `WeightsView`: either uniform weights or a slice of explicit weights.
- `SamplesView`: sample-major `[n_samples, n_features]` view, produced by `buffer_samples`.

Multi-output targets:

- `n_outputs = 1` for regression and binary classification.
- `n_outputs = K` for K-class classification (one row per class/logit).

## Construction

### `Dataset::from_array`

Convenience constructor for dense feature-major input:

- Features shape: `[n_features, n_samples]`
- Targets shape (optional): `[n_outputs, n_samples]`
- Weights shape (optional): `[n_samples]`

### `Dataset::builder`

Use the builder for mixed dense/sparse and explicit metadata:

```rust
use boosters::data::Dataset;
use ndarray::array;

let ds = Dataset::builder()
    .add_feature("age", array![25.0, 30.0, 35.0])
    .add_categorical("color", array![0.0, 1.0, 2.0])
    .add_sparse("rare", vec![1, 3], vec![10.0, 30.0], 5, 0.0)
    .targets_1d(array![0.0, 1.0, 0.0, 1.0, 0.0].view())
    .build()?;
```

Validation happens in `build()` (empty features, shape mismatches, sparse index invariants).

## Integration and Ownership

The split between raw and binned data is intentional:

- `Dataset` owns raw values and schema.
- `BinnedDataset::from_dataset(&Dataset, &BinningConfig)` derives training bins without retaining a reference.

What needs what:

| Component | Needs bins | Needs raw | Needs targets | Needs weights |
| --- | --- | --- | --- | --- |
| GBDT histogram building | ✅ | ❌ | ❌ (uses grads) | ❌ |
| GBDT split finding | ✅ | ❌ | ❌ | ❌ |
| GBDT prediction | ❌ | ✅ | ❌ | ❌ |
| GBLinear training | ❌ | ✅ | ✅ | ✅ |
| Linear leaves (per-leaf fit) | ❌ | ✅ | ✅ | ✅ |
| Tree SHAP / Linear SHAP | ❌ | ✅ | ❌ | ❌ |

## Files

| Path | Contents |
| ---- | -------- |
| `crates/boosters/src/data/raw/dataset.rs` | `Dataset`, `DatasetBuilder`, access primitives |
| `crates/boosters/src/data/raw/feature.rs` | `Feature` storage (dense/sparse) |
| `crates/boosters/src/data/raw/schema.rs` | `DatasetSchema`, `FeatureMeta`, `FeatureType` |
| `crates/boosters/src/data/raw/views.rs` | `TargetsView`, `WeightsView`, `SamplesView` |

## Design Decisions

**DD-1: Separate raw from binned.** `Dataset` holds raw values; `BinnedDataset` holds bins only.
This keeps ownership clear and avoids raw-value duplication.

**DD-2: Sparse via enum (not traits).** Storage is represented as `Feature::{Dense,Sparse}`.
Matching once avoids per-element branching.

**DD-3: Caller-owned buffers.** Sample-major traversal is done via `buffer_samples()` into a caller buffer.
This enables per-thread buffer reuse without allocations.

**DD-4: Small set of access primitives.** The API exposes a few operations that map to actual algorithms
(scan, dense stream, gather, sample blocks), rather than a large general-purpose accessor surface.

## Testing Strategy

| Category | Tests |
| -------- | ----- |
| Construction | Empty dataset, shape mismatch, mixed dense/sparse |
| Sparse invariants | Unsorted indices, duplicates, out-of-bounds |
| Access primitives | Scan semantics, dense stream semantics, sorted gather requirement |
| Block buffering | Dense/sparse buffering and last-partial-block correctness |
