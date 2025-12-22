//! Parquet dataset loader.
//!
//! See RFC-0029 for design rationale. Parquet support reuses the Arrow loader
//! after reading Parquet into Arrow RecordBatches.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use ndarray::Array2;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::error::DatasetLoadError;
use crate::data::Dataset;

// =============================================================================
// Public API
// =============================================================================

/// Load a Parquet file into a [`Dataset`].
pub fn load_parquet_to_dataset(path: impl AsRef<Path>) -> Result<Dataset, DatasetLoadError> {
	let (batches, schema) = read_parquet_file(path)?;
	let loaded = super::record_batches::LoadedBatches::new(schema, batches)?;
	loaded.to_dataset()
}

/// Load a Parquet file into a row-major Array2<f32> with shape (n_samples, n_features).
pub fn load_parquet_to_row_matrix_f32(path: impl AsRef<Path>) -> Result<Array2<f32>, DatasetLoadError> {
	let (batches, schema) = read_parquet_file(path)?;
	let loaded = super::record_batches::LoadedBatches::new(schema, batches)?;
	loaded.to_row_matrix_f32()
}

/// Load a Parquet file into a column-major Array2<f32> with shape (n_features, n_samples).
pub fn load_parquet_to_col_matrix_f32(path: impl AsRef<Path>) -> Result<Array2<f32>, DatasetLoadError> {
	let (batches, schema) = read_parquet_file(path)?;
	let loaded = super::record_batches::LoadedBatches::new(schema, batches)?;
	loaded.to_col_matrix_f32()
}

/// Load a Parquet file and return both features and targets as vecs.
///
/// Returns `(features_row_major, targets, rows, cols)`.
pub fn load_parquet_xy_row_major_f32(
	path: impl AsRef<Path>,
) -> Result<(Vec<f32>, Vec<f32>, usize, usize), DatasetLoadError> {
	let (batches, schema) = read_parquet_file(path)?;
	let loaded = super::record_batches::LoadedBatches::new(schema, batches)?;
	loaded.to_raw_f32()
}

/// Backward-compatible alias for [`load_parquet_xy_row_major_f32`].
///
/// Note: despite the generic name, this returns **row-major** features.
pub fn load_parquet_raw_f32(
	path: impl AsRef<Path>,
) -> Result<(Vec<f32>, Vec<f32>, usize, usize), DatasetLoadError> {
	load_parquet_xy_row_major_f32(path)
}

// =============================================================================
// Internal helpers
// =============================================================================

fn read_parquet_file(path: impl AsRef<Path>) -> Result<(Vec<RecordBatch>, Arc<Schema>), DatasetLoadError> {
	let file = File::open(path.as_ref())?;
	let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
	let schema = builder.schema().clone();
	let reader = builder.build()?;
	let batches: Result<Vec<_>, _> = reader.collect();
	Ok((batches?, schema))
}
