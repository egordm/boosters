//! Dataset and matrix I/O utilities.
//!
//! This module provides loaders for tabular data files.
//! See RFC-0029 for design rationale.
//!
//! # Feature gates
//!
//! - `io-arrow`: Arrow IPC (Feather) file loading
//! - `io-parquet`: Parquet file loading (also enables `io-arrow`)

#[cfg(feature = "io-arrow")]
pub mod arrow;

#[cfg(feature = "io-parquet")]
pub mod parquet;

#[cfg(any(feature = "io-arrow", feature = "io-parquet"))]
mod record_batches;

#[cfg(any(feature = "io-arrow", feature = "io-parquet"))]
mod error;

#[cfg(any(feature = "io-arrow", feature = "io-parquet"))]
pub use error::DatasetLoadError;
