//! Training infrastructure for gradient boosting.
//!
//! This module provides the core types needed for training:
//!
//! ## Shared Infrastructure
//!
//! - [`Gradients`]: Interleaved gradient storage
//! - [`Objective`]: Enum of loss functions for computing gradients
//! - [`Metric`]: Enum of evaluation metrics during training
//! - [`EarlyStopping`]: Callback for stopping when validation metric plateaus
//! - [`TrainingLogger`], [`Verbosity`]: Structured logging
//!
//! ## Model-Specific Training
//!
//! - [`gbdt`]: GBDT (decision tree) training with histogram-based approach
//! - [`gblinear`]: GBLinear training via coordinate descent
//!
//! ## Objectives (Loss Functions)
//!
//! Use enum variants like `Objective::SquaredLoss`, `Objective::LogisticLoss`, etc.
//!
//! Regression:
//! - [`Objective::SquaredLoss`]: Squared error for regression (L2)
//! - [`Objective::AbsoluteLoss`]: Mean absolute error (L1)
//! - [`Objective::PinballLoss`]: Quantile regression (single or multiple quantiles)
//! - [`Objective::PseudoHuberLoss`]: Robust regression, smooth approximation of Huber
//! - [`Objective::PoissonLoss`]: Count data regression
//!
//! Classification:
//! - [`Objective::LogisticLoss`]: Binary cross-entropy
//! - [`Objective::HingeLoss`]: SVM-style binary classification
//! - [`Objective::SoftmaxLoss`]: Multiclass cross-entropy
//!
//! Ranking:
//!
//! Custom: Use [`CustomObjective`] for user-defined objectives.
//!
//! ## Metrics
//!
//! Use `Metric::Rmse`, `Metric::LogLoss`, etc. to construct.
//!
//! - Regression: [`Metric::Rmse`], [`Metric::Mae`], [`Metric::Mape`]
//! - Binary classification: [`Metric::LogLoss`], [`Metric::Auc`], [`Metric::Accuracy`]
//! - Multiclass: [`Metric::MulticlassLogLoss`], [`Metric::MulticlassAccuracy`]
//! - Quantile: [`Metric::Quantile`]
//!
//! Custom: Use [`CustomMetric`] for user-defined metrics.
//!
//! See RFC-0005 for design rationale.

mod callback;
mod eval;
pub mod gbdt;
pub mod gblinear;
mod gradients;
mod logger;
mod metrics;
mod objectives;
pub mod sampling;

// Re-export shared types at the training module level
pub use callback::{EarlyStopAction, EarlyStopping};
pub use eval::{Evaluator, MetricValue};
pub use gradients::{Gradients, GradsTuple};
pub use logger::{TrainingLogger, Verbosity};
pub use metrics::{CustomMetric, CustomMetricFn, Metric, default_metric_for_objective};
pub use objectives::{BaseScoreFn, CustomObjective, GradientFn, Objective};

// Re-export gbdt trainer and params
pub use gbdt::{GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, LinearLeafConfig};

// Re-export sampling types
pub use sampling::{ColSamplingParams, RowSamplingParams};

// Re-export gblinear trainer and params
pub use gblinear::{GBLinearParams, GBLinearTrainError, GBLinearTrainer, UpdateStrategy};
