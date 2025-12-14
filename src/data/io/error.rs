//! Shared error types for dataset I/O.

use std::io;

/// Errors that can occur when loading a dataset.
#[derive(Debug, thiserror::Error)]
pub enum DatasetLoadError {
	#[error("I/O error: {0}")]
	Io(#[from] io::Error),

	#[error("Arrow error: {0}")]
	Arrow(#[from] arrow::error::ArrowError),

	#[cfg(feature = "io-parquet")]
	#[error("Parquet error: {0}")]
	Parquet(#[from] parquet::errors::ParquetError),

	#[error("schema validation failed: {0}")]
	Schema(String),

	#[error("missing required column: {0}")]
	MissingColumn(String),

	#[error("unsupported column type for {column}: expected {expected}, got {got}")]
	UnsupportedType {
		column: String,
		expected: String,
		got: String,
	},

	#[error("row count mismatch: {0}")]
	RowCountMismatch(String),

	#[error("invalid metadata: {0}")]
	InvalidMetadata(String),
}
