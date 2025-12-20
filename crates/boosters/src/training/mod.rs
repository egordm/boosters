//! Training infrastructure for gradient boosting.
//!
//! This module provides the core types needed for training:
//!
//! ## Shared Infrastructure
//!
//! - [`Gradients`]: Interleaved gradient storage
//! - [`Objective`]: Trait for computing gradients (supports single and multi-output)
//! - [`Metric`], [`EvalSet`]: Evaluation during training
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
//! Regression:
//! - [`SquaredLoss`]: Squared error for regression (L2)
//! - [`AbsoluteLoss`]: Mean absolute error (L1)
//! - [`PinballLoss`]: Quantile regression (single or multiple quantiles)
//! - [`PseudoHuberLoss`]: Robust regression, smooth approximation of Huber
//! - [`PoissonLoss`]: Count data regression
//!
//! Classification:
//! - [`LogisticLoss`]: Binary cross-entropy
//! - [`HingeLoss`]: SVM-style binary classification
//! - [`SoftmaxLoss`]: Multiclass cross-entropy
//!
//! Ranking:
//! - [`LambdaRankLoss`]: LambdaMART for learning to rank
//!
//! ## Metrics
//!
//! - [`Rmse`], [`Mae`], [`Mape`]: Regression metrics
//! - [`LogLoss`], [`Auc`], [`Accuracy`]: Binary classification metrics
//! - [`MulticlassLogLoss`], [`MulticlassAccuracy`]: Multiclass metrics
//! - [`QuantileMetric`]: Pinball loss metric for quantile regression
//!
//! See RFC-0009 for design rationale.

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
pub use callback::{EarlyStopping, EarlyStopAction};
pub use eval::{EvalSet, Evaluator, MetricValue};
pub use gradients::{GradsTuple, Gradients};
pub use logger::{TrainingLogger, Verbosity};
pub use metrics::{
    Accuracy, Auc, HuberMetric, LogLoss, Mae, Mape, MarginAccuracy, Metric, MetricFn, MetricKind,
    MulticlassAccuracy, MulticlassLogLoss, PoissonDeviance, QuantileMetric, Rmse,
};
pub use objectives::{
    AbsoluteLoss, HingeLoss, LambdaRankLoss, LogisticLoss, Objective, ObjectiveFn, ObjectiveFnExt,
    PinballLoss, PoissonLoss, PseudoHuberLoss, SoftmaxLoss, SquaredLoss,
    TargetSchema, TaskKind,
};

// Re-export gbdt trainer and params
pub use gbdt::{GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, LinearLeafConfig};

// Re-export sampling types
pub use sampling::{ColSamplingParams, RowSamplingParams};

// Re-export gblinear trainer and params
pub use gblinear::{GBLinearParams, GBLinearTrainer};

