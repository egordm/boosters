# RFC-0013: Arrow and Parquet I/O

- **Status**: Implemented
- **Created**: 2024-12-15
- **Updated**: 2025-01-21
- **Scope**: Data loading from Arrow and Parquet files

## Summary

Optional data loading from Arrow IPC (Feather) and Parquet files into boosters matrix and dataset types, with support for two schema layouts and automatic missing value handling.

## Design

### Feature Flags

- `io-arrow`: Enables Arrow IPC file loading. Depends on `arrow` crate.
- `io-parquet`: Enables Parquet file loading. Depends on `parquet` and `arrow` crates.

Both features share common RecordBatch conversion logic in `record_batches.rs`.

### Supported Schemas

Two column layouts are supported:

1. **Wide columns**: Individual feature columns named `x_0`, `x_1`, ..., `x_{d-1}` (Float32)
2. **FixedSizeList**: Single `x` column of type `FixedSizeList<Float32, d>`

Required columns:
- `y` (Float32 or Int32): Target values

The loader auto-detects the schema by checking for a `FixedSizeList` column named `x` first, then falling back to wide column enumeration.

### Arrow Integration

Core conversion is handled by `LoadedBatches` which wraps schema + batches:

```rust
pub(super) struct LoadedBatches {
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
    rows: usize,
    cols: usize,
}
```

Conversion methods:
- `to_dataset()` → `Dataset` (feature columns + targets)
- `to_row_matrix_f32()` → `SamplesView<f32>` (features only)
- `to_col_matrix_f32()` → `FeaturesView<f32>` (features only)
- `to_raw_f32()` → `(Vec<f32>, Vec<f32>, rows, cols)` (row-major features + targets)

### Missing Value Handling

Null values in Arrow arrays are converted to `f32::NAN`:

```rust
arr.iter().map(|v| v.unwrap_or(f32::NAN))
```

Int32 columns (common for labels) are cast to f32 with null → 0.

### Parquet Loading

Parquet files are read into Arrow RecordBatches using `ParquetRecordBatchReaderBuilder`, then processed through the same `LoadedBatches` pipeline:

```rust
fn read_parquet_file(path: impl AsRef<Path>) -> Result<(Vec<RecordBatch>, Arc<Schema>), DatasetLoadError> {
    let file = File::open(path.as_ref())?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let reader = builder.build()?;
    let batches: Result<Vec<_>, _> = reader.collect();
    Ok((batches?, schema))
}
```

## Key Functions

### Arrow IPC (`src/data/io/arrow.rs`)

| Function | Returns |
|----------|---------|
| `load_ipc_to_dataset(path)` | `Dataset` |
| `load_ipc_to_row_matrix_f32(path)` | `SamplesView<f32>` |
| `load_ipc_to_col_matrix_f32(path)` | `FeaturesView<f32>` |
| `load_ipc_xy_row_major_f32(path)` | `(Vec<f32>, Vec<f32>, rows, cols)` |
| `load_ipc_raw_f32(path)` | Alias for above |

### Parquet (`src/data/io/parquet.rs`)

| Function | Returns |
|----------|---------|
| `load_parquet_to_dataset(path)` | `Dataset` |
| `load_parquet_to_row_matrix_f32(path)` | `SamplesView<f32>` |
| `load_parquet_to_col_matrix_f32(path)` | `FeaturesView<f32>` |
| `load_parquet_xy_row_major_f32(path)` | `(Vec<f32>, Vec<f32>, rows, cols)` |
| `load_parquet_raw_f32(path)` | Alias for above |

## Error Handling

All functions return `Result<T, DatasetLoadError>` with variants:

- `Io` - File I/O errors
- `Arrow` - Arrow format errors
- `Parquet` - Parquet format errors (io-parquet only)
- `Schema` - Schema validation failures
- `MissingColumn` - Required column not found
- `UnsupportedType` - Column type mismatch

## Dependencies

```toml
arrow = { version = "54", optional = true, default-features = false, features = ["ipc"] }
parquet = { version = "54", optional = true, default-features = false, features = ["arrow", "snap"] }
```

## Future Considerations

**Deprecation after Python bindings**: Once Python bindings are implemented with NumPy/Pandas zero-copy support, Arrow/Parquet I/O may be deprecated or moved to dev-only. Most users will interact through Python, making native data loading less relevant. The features may remain useful for:

- Rust-only usage
- Internal benchmarking (quality benchmarks currently use Parquet)
- Testing data fixtures

## Changelog

- 2025-01-21: Updated terminology (SamplesView, FeaturesView) to match codebase conventions
