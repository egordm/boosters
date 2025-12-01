//! Training infrastructure for gradient boosting.
//!
//! This module provides the core types needed for training:
//!
//! - [`GradientPair`]: Gradient and hessian pair for optimization
//! - [`Loss`]: Trait for computing gradients from predictions and labels
//! - [`MulticlassLoss`]: Trait for multiclass gradient computation
//! - [`Metric`]: Trait for evaluating model quality
//! - [`EarlyStopping`]: Callback for stopping training when validation metric plateaus
//! - [`TrainingLogger`]: Structured logging with verbosity levels
//!
//! ## Loss Functions
//!
//! - [`SquaredLoss`]: Squared error for regression
//! - [`LogisticLoss`]: Binary cross-entropy for classification
//! - [`SoftmaxLoss`]: Multiclass cross-entropy
//! - [`QuantileLoss`]: Pinball loss for quantile regression
//!
//! ## Metrics
//!
//! - [`Rmse`]: Root mean squared error
//! - [`Mae`]: Mean absolute error
//! - [`LogLoss`]: Binary cross-entropy
//! - [`Accuracy`]: Classification accuracy
//! - [`Auc`]: Area under ROC curve
//!
//! See RFC-0009 for design rationale.

mod callback;
mod gradient;
mod logger;
mod loss;
mod metric;

pub use callback::EarlyStopping;
pub use gradient::GradientPair;
pub use logger::{TrainingLogger, Verbosity};
pub use loss::{LogisticLoss, Loss, MulticlassLoss, QuantileLoss, SoftmaxLoss, SquaredLoss};
pub use metric::{Accuracy, Auc, LogLoss, Mae, Metric, MulticlassAccuracy, Rmse};
