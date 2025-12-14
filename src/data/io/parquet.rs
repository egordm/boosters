//! Parquet dataset loader.
//!
//! See RFC-0029 for design rationale. Parquet support reuses the Arrow loader
//! after reading Parquet into Arrow RecordBatches.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::error::DatasetLoadError;
use crate::data::{ColMatrix, Dataset, RowMatrix};

// =============================================================================
// Public API
// =============================================================================

/// Load a Parquet file into a [`Dataset`].
pub fn load_parquet_to_dataset(path: impl AsRef<Path>) -> Result<Dataset, DatasetLoadError> {
	let (batches, schema) = read_parquet_file(path)?;
	let loaded = super::record_batches::LoadedBatches::new(schema, batches)?;
	loaded.to_dataset()
}

/// Load a Parquet file into a row-major [`RowMatrix`].
pub fn load_parquet_to_row_matrix_f32(path: impl AsRef<Path>) -> Result<RowMatrix<f32>, DatasetLoadError> {
	let (batches, schema) = read_parquet_file(path)?;
	let loaded = super::record_batches::LoadedBatches::new(schema, batches)?;
	loaded.to_row_matrix_f32()
}

/// Load a Parquet file into a column-major [`ColMatrix`].
pub fn load_parquet_to_col_matrix_f32(path: impl AsRef<Path>) -> Result<ColMatrix<f32>, DatasetLoadError> {
	let (batches, schema) = read_parquet_file(path)?;
	let loaded = super::record_batches::LoadedBatches::new(schema, batches)?;
	loaded.to_col_matrix_f32()
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
