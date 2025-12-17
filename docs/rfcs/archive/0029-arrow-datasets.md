# RFC-0029: Arrow/Parquet I/O for `Dataset` and `DenseMatrix`

- **Status**: Draft
- **Created**: 2025-12-14
- **Updated**: 2025-12-14
- **Depends on**: RFC-0004 (DMatrix), RFC-0007 (Serialization), RFC-0010 (Matrix layouts)
- **Scope**: General-purpose dataset/matrix loading (library feature) + reuse in tests/benches/tools

## Summary

Add first-class support for loading tabular datasets and dense matrices from Apache Arrow formats.

This provides a single, general-purpose interface that can be used by:

- library users (loading datasets in Rust)
- tests (fixtures without bespoke formats)
- benchmarks and quality harnesses (consistent, reproducible inputs)
- future Python bindings (natural interoperability via Arrow)

## Motivation

We need a way to store and load datasets for:

- cross-library comparisons (XGBoost/LightGBM/booste-rs)
- “quality harness” evaluation (RMSE/MAE/logloss/accuracy/AUC)
- core library usage (users want to load data without writing converters)
- future Python bindings (Arrow is a natural bridge)

Requirements:

- compact enough to avoid repo bloat (or support download-and-cache)
- fast to load and convert to existing in-memory types (`Dataset`, `DenseMatrix`)
- easy to generate from Python tooling (pyarrow / pandas)
- schema + metadata stable and versioned

Arrow IPC (Feather) and Parquet are widely supported standards that fit these constraints.

We choose Arrow specifically because it integrates well with modern data formats and pipelines, and because it makes later Python bindings easier.

## Design

### Overview

We introduce an Arrow/Parquet I/O layer that can:

1. Load a dataset/matrix from disk (Arrow IPC file, and Parquet).
2. Validate schema + metadata.
3. Convert into core in-memory representations:
    - `Dataset` (feature columns + target)
    - `DenseMatrix` (row-major or column-major), and existing aliases (`RowMatrix`, `ColMatrix`)

This is intended to be a **major library feature**, but still feature-gated so default builds remain lightweight.

The I/O features must be **off by default** initially to avoid increasing compile times and dependency surface for users who don’t need file loading. Once we add Python bindings, we may revisit defaults.

### Module placement

The loader should live under the main data module, not under testing:

- `booste_rs::data::io::arrow` (Arrow IPC)
- `booste_rs::data::io::parquet` (Parquet)

Tests and benchmarks should call these same loaders (possibly via thin wrappers in `booste_rs::testing`).

### Data Structures

```rust
/// Lightweight metadata describing an on-disk dataset.
#[derive(Debug, Clone)]
pub struct DatasetMeta {
    pub rows: usize,
    pub cols: usize,
    pub task: DatasetTask,
    pub n_classes: Option<usize>,
    pub split_seed: Option<u64>,
    pub schema_version: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetTask {
    Regression,
    Binary,
    Multiclass,
}

/// Loaded dense dataset (numeric) ready to convert to booste-rs matrices.
#[derive(Debug, Clone)]
pub struct ArrowDatasetF32 {
    pub meta: DatasetMeta,
    /// Row-major dense features (materialized once).
    pub x_row_major: Vec<f32>,
    pub y: Vec<f32>,
}

/// Result of loading a dataset in column-oriented form suitable for `Dataset`.
///
/// This is a better match for Arrow’s native columnar representation.
#[derive(Debug, Clone)]
pub struct ArrowColumnsF32 {
    pub meta: DatasetMeta,
    pub features: Vec<crate::data::FeatureColumn>,
    pub targets: Vec<f32>,
}
```

### Schema

We support two schema representations to balance ergonomics (Python) and compactness (wide datasets):

- **Schema A (wide columns)**: `x_0 .. x_{d-1}` as `Float32`, plus `y`.
- **Schema B (single column)**: one `FixedSizeList<Float32>` column `x` (length = `cols`), plus `y`.

Optional columns (future):

- `w`: per-row weight (`Float32`)
- categorical columns: either `Int32` categories, or Arrow dictionary encoding

Metadata (stored as Arrow schema metadata key-value pairs):

- `booste_rs.schema_version` = `1`
- `booste_rs.task` = `regression|binary|multiclass`
- `booste_rs.n_classes` (optional)
- `booste_rs.split_seed` (optional)
- (optional) `booste_rs.split.train`, `booste_rs.split.valid`, `booste_rs.split.test` (see DD-2)

Parquet stores metadata differently, but supports key-value metadata as well; we use the same keys.

### Algorithms

Loading:

1. Read Arrow IPC (or Parquet).
2. Validate: required columns, dtype, row count, metadata.
3. Convert once into the requested in-memory representation (outside timed regions):
    - `Dataset`: build `FeatureColumn::Numeric { values: Vec<f32> }` per feature
    - `DenseMatrix`: build either `RowMajor` or `ColMajor` buffer

Conversion:

- For prediction benchmarks: prefer `RowMatrix` for contiguous row slices.
- For GBLinear training: prefer direct conversion into `ColMatrix` since Arrow is columnar.
- For tree training: either route is fine; `Dataset` may be the most ergonomic since it already models “feature columns + targets”.

### API

Proposed module (feature-gated):

```rust
pub mod data {
    pub mod io {
        #[cfg(feature = "io-arrow")]
        pub mod arrow {
            pub fn load_ipc_to_dataset(
                path: impl AsRef<std::path::Path>,
            ) -> Result<crate::data::Dataset, DatasetLoadError>;

            pub fn load_ipc_to_row_matrix_f32(
                path: impl AsRef<std::path::Path>,
            ) -> Result<crate::data::RowMatrix<f32>, DatasetLoadError>;

            pub fn load_ipc_to_col_matrix_f32(
                path: impl AsRef<std::path::Path>,
            ) -> Result<crate::data::ColMatrix<f32>, DatasetLoadError>;
        }

        #[cfg(feature = "io-parquet")]
        pub mod parquet {
            pub fn load_parquet_to_dataset(
                path: impl AsRef<std::path::Path>,
            ) -> Result<crate::data::Dataset, DatasetLoadError>;
        }
    }
}
```

This makes Arrow/Parquet loading a reusable building block for tests and benchmarks without requiring a bespoke testing-only format.

## Design Decisions

### DD-1: Support multiple schemas (wide columns + FixedSizeList)

**Context**: Arrow has multiple ways to represent a dense matrix. Python ergonomics and wide datasets push in different directions.

**Options considered**:

1. Wide columns (`x_0..x_{d-1}`)
2. One `FixedSizeList<Float32>` column

**Decision**: Support both.

- Wide columns are the most ergonomic for Python/pandas and map directly to `Dataset` (feature columns).
- FixedSizeList keeps schema compact for very wide datasets and is attractive for certain pipelines.

**Consequences**:

- The loader is slightly more complex, but interoperability improves.
- Tests/benchmarks can accept either shape depending on how the data was generated.

### DD-2: How to store splits

**Context**: We need deterministic train/valid/test splits.

**Options considered**:

1. Store splits as separate Arrow files (`train.arrow`, `valid.arrow`, `test.arrow`).
2. Store split indices in metadata.

**Decision**: Prefer **separate files** for simplicity and streaming friendliness.

**Consequences**:

- The “dataset” becomes a small directory with multiple IPC files.
- Avoids giant metadata payloads.

## Integration

| Component | Integration Point | Notes |
|-----------|------------------|-------|
| `Dataset` | Loader can construct `Dataset` directly | Mirrors Arrow’s columnar structure |
| `DenseMatrix` | Loader can construct `RowMajor` or `ColMajor` buffers | Works with existing matrix APIs |
| RFC-0004 (DMatrix) | Loader converts into `RowMatrix`/`ColMatrix` | Keeps DMatrix abstractions intact |
| Bench suites | Data loaded once, timed regions operate in-memory | Prevents IO-in-timed-region mistakes |
| tools/data_generation | Emits Arrow IPC datasets with schema+metadata | Keeps Python as the dataset source |
| Future Python bindings | Arrow acts as a stable interchange boundary | Avoids bespoke FFI formats |

## Open Questions

1. Should Parquet be MVP, or follow after Arrow IPC?
2. Should we standardize on `f32` for all loaders and convert for external libraries, or allow `f64` loads?
3. Do we want to support categorical columns in MVP (int32) or postpone?
4. Do we want a stable dataset directory layout (`train.arrow`, `valid.arrow`, `test.arrow`) as part of this RFC?

## Future Work

- [ ] Implement `io-arrow` feature with Arrow IPC loader
- [ ] Implement `io-parquet` feature (optional)
- [ ] Add small test fixtures (tiny Arrow IPC files) for loader correctness
- [ ] Update benchmark plan + tools to emit Arrow datasets in a consistent schema
- [ ] Keep the loader APIs stable for future Python binding integration

## References

- Research: ../research/arrow-datasets.md
- Apache Arrow format docs: <https://arrow.apache.org/>

## Changelog

- 2025-12-14: Initial draft
