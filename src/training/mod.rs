//! Training infrastructure for gradient boosting.
//!
//! This module provides the core types needed for training:
//!
//! ## Shared Infrastructure
//!
//! - [`GradientBuffer`]: Structure-of-Arrays gradient storage
//! - [`Loss`], [`MulticlassLoss`]: Traits for computing gradients
//! - [`Metric`], [`EvalSet`]: Evaluation during training
//! - [`EarlyStopping`]: Callback for stopping when validation metric plateaus
//! - [`TrainingLogger`], [`Verbosity`]: Structured logging
//!
//! ## Model-Specific Training
//!
//! - [`gbtree`]: GBTree (decision tree) training with histogram-based approach
//! - [`linear`]: GBLinear training via coordinate descent
//!
//! ## Loss Functions
//!
//! Single-output (regression, binary classification):
//! - [`SquaredLoss`]: Squared error for regression
//! - [`LogisticLoss`]: Binary cross-entropy for classification
//! - [`PseudoHuberLoss`]: Robust regression, smooth approximation of Huber
//! - [`HingeLoss`]: SVM-style binary classification
//!
//! Multi-output (multiclass, multi-quantile):
//! - [`SoftmaxLoss`]: Multiclass cross-entropy
//! - [`QuantileLoss`]: Pinball loss (single or multiple quantiles)
//!
//! ## Metrics
//!
//! - [`Rmse`], [`Mae`], [`Mape`]: Regression metrics
//! - [`LogLoss`], [`Auc`], [`Accuracy`]: Binary classification metrics
//! - [`MulticlassLogLoss`], [`MulticlassAccuracy`]: Multiclass metrics
//! - [`QuantileMetric`]: Pinball loss metric for quantile regression
//!
//! See RFC-0009 for design rationale.

mod buffer;
mod callback;
pub mod gbtree;
pub mod linear;
mod logger;
mod loss;
mod metric;

// Re-export shared types at the training module level
pub use buffer::GradientBuffer;
pub use callback::EarlyStopping;
pub use logger::{TrainingLogger, Verbosity};
pub use loss::{
    HingeLoss, LogisticLoss, Loss, MulticlassLoss, PseudoHuberLoss, QuantileLoss, SoftmaxLoss,
    SquaredLoss,
};
pub use metric::{
    Accuracy, Auc, EvalMetric, EvalSet, LogLoss, Mae, Mape, Metric, MulticlassAccuracy,
    MulticlassLogLoss, QuantileLoss as QuantileMetric, Rmse, SimpleMetric,
};

// Re-export gbtree types for convenience (commonly used)
pub use gbtree::{
    BaseScore, BinCuts, BinIndex, BuildingNode, BuildingTree, ChildSide, CutFinder,
    DepthWisePolicy, DepthWiseState, ExactQuantileCuts, FeatureHistogram, GBTreeTrainer,
    GainParams, GreedySplitFinder, GrowthPolicy, GrowthState, GrowthStrategy, HistogramBuilder,
    HistogramSubtractor, LeafWisePolicy, LeafWiseState, NodeCandidate, NodeHistogram,
    QuantizedEvalSet, QuantizedMatrix, Quantizer, RowPartitioner, SplitFinder, SplitInfo,
    TrainerParams, TreeGrower, TreeParams, leaf_objective, leaf_weight, split_gain,
};

// Re-export linear types for convenience
pub use linear::{
    CyclicSelector, FeatureSelector, FeatureSelectorKind, GreedySelector, LinearTrainer,
    LinearTrainerConfig, RandomSelector, SelectorState, ShuffleSelector, ThriftySelector,
    UpdateConfig, UpdaterKind, update_bias,
};
