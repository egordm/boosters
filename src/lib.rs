//! booste-rs: A gradient boosting library for Rust.
//!
//! This crate provides native Rust implementations for gradient boosted decision trees,
//! with support for loading models from external frameworks like XGBoost and LightGBM.
//!
//! # Module Structure
//!
//! - [`data`]: Data matrix types and dataset utilities
//! - [`inference`]: Prediction infrastructure (trees, forests, linear models)
//! - [`training`]: Training infrastructure (objectives, metrics, trainers)
//! - [`compat`]: External model loading (XGBoost, LightGBM)
//!
//! # Quick Start
//!
//! ```ignore
//! use booste_rs::{
//!     data::BinnedDatasetBuilder,
//!     training::{GBDTTrainer, GBDTParams, SquaredLoss},
//! };
//!
//! // Prepare data
//! let dataset = BinnedDatasetBuilder::new(&features)
//!     .max_bins(256)
//!     .build()
//!     .unwrap();
//!
//! // Train model
//! let trainer = GBDTTrainer::new(SquaredLoss, GBDTParams::default());
//! let forest = trainer.train(&dataset, &targets, &[]);
//!
//! // Predict using forest directly
//! let prediction = forest.predict_row(&test_row);
//! ```

// Re-export approx traits for users who want to compare predictions
pub use approx;

pub mod compat;
pub mod data;
pub mod inference;
pub mod testing;
pub mod training;
pub mod utils;
