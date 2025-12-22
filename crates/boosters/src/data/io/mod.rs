//! Dataset and matrix I/O utilities.
//!
//! This module provides loaders for tabular data files.
//! See RFC-0029 for design rationale.
//!
//! # Feature gates
//!
//! - `io-parquet`: Parquet file loading

#[cfg(feature = "io-parquet")]
pub mod parquet;

#[cfg(feature = "io-parquet")]
mod record_batches;

#[cfg(feature = "io-parquet")]
mod error;

#[cfg(feature = "io-parquet")]
pub use error::DatasetLoadError;
