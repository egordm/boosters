//! Prediction pipeline and visitor patterns.
//!
//! This module provides the prediction infrastructure for traversing trees
//! and accumulating leaf values.
//!
//! # Overview
//!
//! The prediction pipeline uses a visitor pattern:
//!
//! - [`TreeVisitor`]: Core trait for traversing a single tree
//! - [`ScalarVisitor`]: Visitor for forests with scalar leaves
//! - [`Predictor`]: Orchestrates batch prediction over a forest
//!
//! # Output Format
//!
//! Predictions are returned as [`PredictionOutput`], a flat row-major buffer
//! with shape `(num_rows, num_groups)`. For regression, `num_groups = 1`.
//! For multiclass with K classes, `num_groups = K`.
//!
//! See RFC-0003 for design rationale.

mod output;
mod visitor;

pub use output::PredictionOutput;
pub use visitor::{Predictor, ScalarVisitor, TreeVisitor};
