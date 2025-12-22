//! boosters: A gradient boosting library for Rust.
//!
//! This crate provides native Rust implementations for gradient boosted decision trees,
//! with support for loading models from external frameworks like XGBoost and LightGBM.
//!
//! # Module Structure
//!
//! - [`model`]: High-level model types (`GBDTModel`, `GBLinearModel`)
//! - [`data`]: Data matrix types and dataset utilities
//! - [`training`]: Training infrastructure (objectives, metrics, trainers)
//! - [`compat`]: External model loading (XGBoost, LightGBM)
//!
//! # Quick Start: Training (Recommended)
//!
//! Use the high-level model API with configuration builders:
//!
//! ```ignore
//! use boosters::model::GBDTModel;
//! use boosters::model::gbdt::GBDTConfig;
//! use boosters::data::RowMatrix;
//! use boosters::training::{Objective, Metric};
//!
//! // Build configuration
//! let config = GBDTConfig::builder()
//!     .objective(Objective::squared_error())
//!     .metric(Metric::rmse())
//!     .n_trees(100)
//!     .learning_rate(0.1)
//!     .build()?;
//!
//! // Train model
//! let model = GBDTModel::train(&dataset, &targets, &[], config)?;
//!
//! // Predict with DataMatrix
//! let features = RowMatrix::from_vec(data, n_rows, n_features);
//! let predictions = model.predict(&features, None);  // ColMatrix<f32>, sequential
//! let probs = predictions.col_slice(0);              // First output column
//! ```
//!
//! # Advanced: Direct Trainer Access
//!
//! For fine-grained control, use the trainer API directly:
//!
//! ```ignore
//! use boosters::training::{GBDTTrainer, GBDTParams, SquaredLoss, Rmse};
//! use boosters::Parallelism;
//!
//! let params = GBDTParams { n_trees: 100, ..Default::default() };
//! let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);
//! let forest = trainer.train(&binned_dataset, targets.view(), weights.view(), &[], Parallelism::Sequential)?;
//! ```
//!
//! # Loading XGBoost Models
//!
//! ```ignore
//! use boosters::compat::xgboost::XgbModel;
//! use boosters::data::RowMatrix;
//!
//! let model = XgbModel::from_file("model.json")?;
//! let forest = model.to_forest()?;
//!
//! // Use Predictor for efficient batch prediction
//! use boosters::inference::gbdt::Predictor;
//! let features = RowMatrix::from_vec(data, n_rows, n_features);
//! let predictor = Predictor::new(&forest);
//! let predictions = predictor.predict(&features);
//! ```

// Re-export approx traits for users who want to compare predictions
pub use approx;

pub mod compat;
pub mod data;
pub mod explainability;
pub mod inference;
pub mod model;
pub mod repr;
pub mod testing;
pub mod training;
pub mod utils;

// =============================================================================
// Convenience Re-exports
// =============================================================================

// High-level model types
pub use model::{GBDTModel, GBLinearModel, ModelMeta, TaskKind};

// Configuration types (most users want these)
pub use model::gbdt::{GBDTConfig, RegularizationParams, SamplingParams, TreeParams};
pub use model::gblinear::GBLinearConfig;

// Training types (objectives, metrics)
pub use training::{Metric, MetricFn, Objective, ObjectiveFn};

// Data types (for preparing training data)
pub use data::{ColMatrix, Dataset, DenseMatrix, RowMatrix};

// Shared utilities
pub use utils::{Parallelism, run_with_threads};

