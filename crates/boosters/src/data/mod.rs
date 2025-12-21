//! Data input abstractions for feature matrices.
//!
//! This module provides the [`DataMatrix`] trait and implementations for
//! accessing feature data during tree traversal and training.
//!
//! # Overview
//!
//! The core abstraction is [`DataMatrix`], which provides a uniform interface
//! for accessing feature values regardless of the underlying storage format
//! (dense, sparse, Arrow, etc.).
//!
//! # Storage Types
//!
//! - [`DenseMatrix`]: Dense storage with configurable layout (row-major or column-major)
//! - [`binned::BinnedDataset`]: Quantized feature data for GBDT training
//!
//! # Layouts
//!
//! Dense matrices support two memory layouts via [`Layout`]:
//! - [`RowMajor`] (default): Rows are contiguous, optimal for row-based access (inference)
//! - [`ColMajor`]: Columns are contiguous, optimal for column-based access (training)
//!
//! # Missing Values
//!
//! Missing values are represented as `f32::NAN`. This is the modern standard
//! used by XGBoost and other libraries.
//!
//! See RFC-0004 for design rationale, RFC-0010 for layout abstraction.

pub mod binned;
mod dataset;
mod matrix;
mod traits;
mod ndarray;

#[cfg(any(feature = "io-arrow", feature = "io-parquet"))]
pub mod io;

pub use matrix::{ColMajor, DenseColumnIter, DenseMatrix, Layout, RowMajor, StridedIter};
pub use dataset::{Dataset, DatasetError, FeatureColumn};
pub use traits::{DataMatrix, FeatureAccessor, RowView};

pub use ndarray::axes;

// Re-export binned types for convenience
pub use binned::{
    BinMapper, BinStorage, BinType, BinnedDataset, BinnedDatasetBuilder, BinningConfig,
    BinningStrategy, BuildError, FeatureGroup, FeatureMeta, FeatureType, FeatureView,
    GroupLayout, GroupSpec, GroupStrategy, MissingType, RowView as BinnedRowView,
};

/// Type alias for row-major dense matrix (the common case).
///
/// This is equivalent to `DenseMatrix<T, RowMajor>` but with better type inference
/// when working with code that accepts multiple layout variants.
///
/// Use this when you need to explicitly specify a concrete matrix type,
/// especially when passing to generic functions that accept `impl DataMatrix`.
pub type RowMatrix<T = f32, S = Box<[T]>> = DenseMatrix<T, RowMajor, S>;

/// Type alias for column-major dense matrix.
///
/// Use this layout when column iteration is the primary access pattern
/// (e.g., during training where you iterate over features).
pub type ColMatrix<T = f32, S = Box<[T]>> = DenseMatrix<T, ColMajor, S>;
