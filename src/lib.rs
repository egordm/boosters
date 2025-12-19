//! boosters: A gradient boosting library for Rust.
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
//! # Quick Start: Training
//!
//! ```ignore
//! use boosters::data::{BinnedDatasetBuilder, ColMatrix, DenseMatrix, RowMajor};
//! use boosters::training::{GBDTTrainer, GBDTParams, SquaredLoss, Rmse};
//!
//! // Prepare column-major data for training
//! let row_matrix: DenseMatrix<f32, RowMajor> = DenseMatrix::from_vec(features, n_rows, n_cols);
//! let col_matrix: ColMatrix<f32> = row_matrix.to_layout();
//! let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 256).build()?;
//!
//! // Train model
//! let trainer = GBDTTrainer::new(SquaredLoss, Rmse, GBDTParams::default());
//! let forest = trainer.train(&dataset, &targets, &[], &[])?;
//!
//! // Predict
//! let prediction = forest.predict_row(&test_row);
//! ```
//!
//! # Quick Start: Loading XGBoost Models
//!
//! ```ignore
//! use boosters::compat::xgboost::XgbModel;
//!
//! let model = XgbModel::from_file("model.json")?;
//! let forest = model.to_forest()?;
//!
//! let prediction = forest.predict_row(&features);
//! ```

// Re-export approx traits for users who want to compare predictions
pub use approx;

pub mod compat;
pub mod data;
pub mod inference;
pub mod io;
pub mod model;
pub mod repr;
pub mod testing;
pub mod training;
pub mod utils;
