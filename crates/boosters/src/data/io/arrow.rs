//! Arrow IPC (Feather) dataset loader.
//!
//! See RFC-0029 for design rationale.
//!
//! # Supported schemas
//!
//! - **Wide columns**: `x_0`, `x_1`, ..., `x_{d-1}` (Float32) + `y` (Float32)
//! - **FixedSizeList**: single `x` column of type `FixedSizeList<Float32>` + `y` (Float32)
//!
//! Optional columns:
//! - `w`: per-row weight (Float32)
//!

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::Schema;
use arrow::ipc::reader::FileReader;
use arrow::record_batch::RecordBatch;

use super::error::DatasetLoadError;
use crate::data::{ColMatrix, Dataset, RowMatrix};

// =============================================================================
// Public API
// =============================================================================

/// Load an Arrow IPC file into a [`Dataset`].
///
/// Expects either wide columns (`x_0..x_{d-1}`) or a single `x` FixedSizeList column,
/// plus a `y` column for targets.
pub fn load_ipc_to_dataset(path: impl AsRef<Path>) -> Result<Dataset, DatasetLoadError> {
	let (batches, schema) = read_ipc_file(path)?;
	let loaded = super::record_batches::LoadedBatches::new(schema, batches)?;
	loaded.to_dataset()
}

/// Load an Arrow IPC file into a row-major [`RowMatrix`].
///
/// Only loads feature columns (ignores `y` and `w`).
pub fn load_ipc_to_row_matrix_f32(path: impl AsRef<Path>) -> Result<RowMatrix<f32>, DatasetLoadError> {
	let (batches, schema) = read_ipc_file(path)?;
	let loaded = super::record_batches::LoadedBatches::new(schema, batches)?;
	loaded.to_row_matrix_f32()
}

/// Load an Arrow IPC file into a column-major [`ColMatrix`].
///
/// Only loads feature columns (ignores `y` and `w`).
pub fn load_ipc_to_col_matrix_f32(path: impl AsRef<Path>) -> Result<ColMatrix<f32>, DatasetLoadError> {
	let (batches, schema) = read_ipc_file(path)?;
	let loaded = super::record_batches::LoadedBatches::new(schema, batches)?;
	loaded.to_col_matrix_f32()
}

/// Load an Arrow IPC file and return both features and targets as vecs.
///
/// Returns `(features_row_major, targets, rows, cols)`.
pub fn load_ipc_xy_row_major_f32(
	path: impl AsRef<Path>,
) -> Result<(Vec<f32>, Vec<f32>, usize, usize), DatasetLoadError> {
	let (batches, schema) = read_ipc_file(path)?;
	let loaded = super::record_batches::LoadedBatches::new(schema, batches)?;
	loaded.to_raw_f32()
}

/// Backward-compatible alias for [`load_ipc_xy_row_major_f32`].
///
/// Note: despite the generic name, this returns **row-major** features.
pub fn load_ipc_raw_f32(
	path: impl AsRef<Path>,
) -> Result<(Vec<f32>, Vec<f32>, usize, usize), DatasetLoadError> {
	load_ipc_xy_row_major_f32(path)
}

// =============================================================================
// Internal helpers
// =============================================================================

fn read_ipc_file(path: impl AsRef<Path>) -> Result<(Vec<RecordBatch>, Arc<Schema>), DatasetLoadError> {
	let file = File::open(path.as_ref())?;
	let reader = FileReader::try_new(BufReader::new(file), None)?;
	let schema = reader.schema();
	let batches: Result<Vec<_>, _> = reader.collect();
	Ok((batches?, schema))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
	use super::*;
	use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, Float32Builder};
	use arrow::datatypes::DataType;
	use arrow::datatypes::Field;
	use arrow::ipc::writer::FileWriter;
	use std::io::BufWriter;
	use tempfile::NamedTempFile;

	fn create_wide_columns_ipc() -> NamedTempFile {
		// Create schema with x_0, x_1, y columns
		let schema = Arc::new(Schema::new(vec![
			Field::new("x_0", DataType::Float32, true),
			Field::new("x_1", DataType::Float32, true),
			Field::new("y", DataType::Float32, true),
		]));

		let x0 = Float32Array::from(vec![1.0, 2.0, 3.0]);
		let x1 = Float32Array::from(vec![4.0, 5.0, 6.0]);
		let y = Float32Array::from(vec![0.0, 1.0, 0.0]);

		let batch = RecordBatch::try_new(
			schema.clone(),
			vec![
				Arc::new(x0) as ArrayRef,
				Arc::new(x1) as ArrayRef,
				Arc::new(y) as ArrayRef,
			],
		)
		.unwrap();

		let file = NamedTempFile::new().unwrap();
		let writer = BufWriter::new(file.reopen().unwrap());
		let mut ipc_writer = FileWriter::try_new(writer, &schema).unwrap();
		ipc_writer.write(&batch).unwrap();
		ipc_writer.finish().unwrap();

		file
	}

	fn create_fixed_list_ipc() -> NamedTempFile {
		// Create schema with x (FixedSizeList<Float32, 2>), y columns
		let inner_field = Arc::new(Field::new("item", DataType::Float32, true));
		let schema = Arc::new(Schema::new(vec![
			Field::new("x", DataType::FixedSizeList(inner_field, 2), true),
			Field::new("y", DataType::Float32, true),
		]));

		// Build the FixedSizeListArray
		let mut builder = Float32Builder::new();
		// Row 0: [1.0, 4.0], Row 1: [2.0, 5.0], Row 2: [3.0, 6.0]
		builder.append_value(1.0);
		builder.append_value(4.0);
		builder.append_value(2.0);
		builder.append_value(5.0);
		builder.append_value(3.0);
		builder.append_value(6.0);
		let values = builder.finish();

		let x = FixedSizeListArray::new(
			Arc::new(Field::new("item", DataType::Float32, true)),
			2,
			Arc::new(values),
			None,
		);

		let y = Float32Array::from(vec![0.0, 1.0, 0.0]);

		let batch = RecordBatch::try_new(
			schema.clone(),
			vec![Arc::new(x) as ArrayRef, Arc::new(y) as ArrayRef],
		)
		.unwrap();

		let file = NamedTempFile::new().unwrap();
		let writer = BufWriter::new(file.reopen().unwrap());
		let mut ipc_writer = FileWriter::try_new(writer, &schema).unwrap();
		ipc_writer.write(&batch).unwrap();
		ipc_writer.finish().unwrap();

		file
	}

	#[test]
	fn test_load_wide_columns_to_dataset() {
		let file = create_wide_columns_ipc();
		let dataset = load_ipc_to_dataset(file.path()).unwrap();

		assert_eq!(dataset.n_rows(), 3);
		assert_eq!(dataset.n_features(), 2);
		assert_eq!(dataset.targets(), &[0.0, 1.0, 0.0]);
	}

	#[test]
	fn test_load_wide_columns_to_row_matrix() {
		let file = create_wide_columns_ipc();
		let matrix = load_ipc_to_row_matrix_f32(file.path()).unwrap();

		assert_eq!(matrix.n_rows(), 3);
		assert_eq!(matrix.n_cols(), 2);
		assert_eq!(matrix.row_slice(0), &[1.0, 4.0]);
		assert_eq!(matrix.row_slice(1), &[2.0, 5.0]);
		assert_eq!(matrix.row_slice(2), &[3.0, 6.0]);
	}

	#[test]
	fn test_load_wide_columns_to_col_matrix() {
		let file = create_wide_columns_ipc();
		let matrix = load_ipc_to_col_matrix_f32(file.path()).unwrap();

		assert_eq!(matrix.n_rows(), 3);
		assert_eq!(matrix.n_cols(), 2);
		assert_eq!(matrix.col_slice(0), &[1.0, 2.0, 3.0]);
		assert_eq!(matrix.col_slice(1), &[4.0, 5.0, 6.0]);
	}

	#[test]
	fn test_load_fixed_list_to_dataset() {
		let file = create_fixed_list_ipc();
		let dataset = load_ipc_to_dataset(file.path()).unwrap();

		assert_eq!(dataset.n_rows(), 3);
		assert_eq!(dataset.n_features(), 2);
		assert_eq!(dataset.targets(), &[0.0, 1.0, 0.0]);
	}

	#[test]
	fn test_load_fixed_list_to_row_matrix() {
		let file = create_fixed_list_ipc();
		let matrix = load_ipc_to_row_matrix_f32(file.path()).unwrap();

		assert_eq!(matrix.n_rows(), 3);
		assert_eq!(matrix.n_cols(), 2);
		assert_eq!(matrix.row_slice(0), &[1.0, 4.0]);
		assert_eq!(matrix.row_slice(1), &[2.0, 5.0]);
		assert_eq!(matrix.row_slice(2), &[3.0, 6.0]);
	}

	#[test]
	fn test_load_raw_f32() {
		let file = create_wide_columns_ipc();
		let (features, targets, rows, cols) = load_ipc_raw_f32(file.path()).unwrap();

		assert_eq!(rows, 3);
		assert_eq!(cols, 2);
		assert_eq!(features, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
		assert_eq!(targets, vec![0.0, 1.0, 0.0]);
	}
}
