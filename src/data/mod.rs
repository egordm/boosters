//! Data input abstractions for feature matrices.
//!
//! This module provides the [`DataMatrix`] trait and implementations for
//! accessing feature data during tree traversal.
//!
//! # Overview
//!
//! The core abstraction is [`DataMatrix`], which provides a uniform interface
//! for accessing feature values regardless of the underlying storage format
//! (dense, sparse, Arrow, etc.).
//!
//! # Storage Types
//!
//! - [`DenseMatrix`]: Row-major dense storage, the most common format
//!
//! # Missing Values
//!
//! Missing values are represented as `f32::NAN`. This is the modern standard
//! used by XGBoost and other libraries.
//!
//! See RFC-0004 for design rationale.

mod dense;
mod traits;

pub use dense::DenseMatrix;
pub use traits::{DataMatrix, RowView};
