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
//! - [`BlockPredictor`]: Block-based prediction for improved cache locality
//! - [`UnrolledPredictor`]: Unrolled-layout optimization for batch prediction
//!
//! # Choosing a Predictor
//!
//! - **Single row or small batches (<100 rows)**: Use [`Predictor`]
//! - **Large batches (100+ rows)**: Use [`BlockPredictor`] or [`UnrolledPredictor`]
//! - **Maximum batch performance**: Use [`UnrolledPredictor`] (unrolls top tree levels)
//!
//! # Unroll Depth
//!
//! [`UnrolledPredictor`] uses const-generic depth for compile-time optimization:
//!
//! - [`UnrolledPredictor4`]: 4 levels (15 nodes) - for shallow trees
//! - [`UnrolledPredictor6`]: 6 levels (63 nodes) - default, matches XGBoost
//! - [`UnrolledPredictor8`]: 8 levels (255 nodes) - for deep trees
//!
//! # Output Format
//!
//! Predictions are returned as [`PredictionOutput`], a flat row-major buffer
//! with shape `(num_rows, num_groups)`. For regression, `num_groups = 1`.
//! For multiclass with K classes, `num_groups = K`.
//!
//! See RFC-0003 for design rationale.

mod block;
mod output;
mod unrolled;
mod visitor;

pub use block::{BlockConfig, BlockPredictor, DEFAULT_BLOCK_SIZE};
pub use output::PredictionOutput;
pub use unrolled::{UnrolledPredictor, UnrolledPredictor4, UnrolledPredictor6, UnrolledPredictor8};
pub use visitor::{Predictor, ScalarVisitor, TreeVisitor};

