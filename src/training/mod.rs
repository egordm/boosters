//! Training infrastructure for gradient boosting.
//!
//! This module provides the core types needed for training:
//!
//! - [`GradientBuffer`]: Structure-of-Arrays gradient storage
//! - [`Loss`]: Trait for computing gradients from predictions and labels
//! - [`MulticlassLoss`]: Trait for multiclass gradient computation
//! - [`Metric`]: Trait for evaluating model quality
//! - [`EarlyStopping`]: Callback for stopping training when validation metric plateaus
//! - [`TrainingLogger`]: Structured logging with verbosity levels
//!
//! ## Quantization (RFC-0011)
//!
//! The [`quantize`] module provides histogram-based training infrastructure:
//! - [`BinCuts`][quantize::BinCuts]: Bin boundaries for all features
//! - [`QuantizedMatrix`][quantize::QuantizedMatrix]: Quantized feature storage
//! - [`Quantizer`][quantize::Quantizer]: Transforms raw features to bins
//!
//! ## Gradient Storage
//!
//! Gradients are stored in Structure-of-Arrays (SoA) layout via [`GradientBuffer`]:
//! - Separate `grads[]` and `hess[]` arrays for better cache efficiency
//! - Shape `[n_samples, n_outputs]` for unified single/multi-output handling
//! - More SIMD-friendly memory access patterns
//!
//! ## Loss Functions
//!
//! - [`SquaredLoss`]: Squared error for regression
//! - [`LogisticLoss`]: Binary cross-entropy for classification
//! - [`SoftmaxLoss`]: Multiclass cross-entropy
//! - [`QuantileLoss`][loss::QuantileLoss]: Pinball loss for quantile regression (single or multiple quantiles)
//! - [`PseudoHuberLoss`]: Robust regression, smooth approximation of Huber loss
//! - [`HingeLoss`]: SVM-style binary classification
//!
//! ## Metrics
//!
//! - [`Rmse`]: Root mean squared error
//! - [`Mae`]: Mean absolute error
//! - [`Mape`]: Mean absolute percentage error
//! - [`LogLoss`]: Binary cross-entropy
//! - [`MulticlassLogLoss`]: Multiclass cross-entropy
//! - [`Accuracy`]: Classification accuracy (binary)
//! - [`MulticlassAccuracy`]: Multiclass accuracy
//! - [`Auc`]: Area under ROC curve
//! - [`QuantileLoss`][metric::QuantileLoss]: Pinball loss for quantile regression
//!
//! ## Evaluation Sets
//!
//! - [`EvalSet`]: Named dataset for evaluation during training (for raw data)
//! - [`QuantizedEvalSet`]: Named dataset for evaluation with quantized data (GBTree)
//! - [`SimpleMetric`]: Helper trait for single-output metrics
//!
//! See RFC-0009 for design rationale, RFC-0011 for quantization.

mod buffer;
mod callback;
pub mod histogram;
mod logger;
mod loss;
mod metric;
pub mod partition;
pub mod quantize;
pub mod split;
pub mod trainer;
pub mod tree;

pub use buffer::GradientBuffer;
pub use callback::EarlyStopping;
pub use histogram::{
    ChildSide, FeatureHistogram, HistogramBuilder, HistogramSubtractor, NodeHistogram,
};
pub use logger::{TrainingLogger, Verbosity};
pub use loss::{
    HingeLoss, LogisticLoss, Loss, MulticlassLoss, PseudoHuberLoss, QuantileLoss, SoftmaxLoss,
    SquaredLoss,
};
pub use metric::{
    Accuracy, Auc, EvalMetric, EvalSet, LogLoss, Mae, Mape, Metric, MulticlassAccuracy,
    MulticlassLogLoss, QuantileLoss as QuantileMetric, Rmse, SimpleMetric,
};
pub use partition::RowPartitioner;
pub use quantize::{BinCuts, BinIndex, QuantizedMatrix, Quantizer};
pub use split::{
    GainParams, GreedySplitFinder, SplitFinder, SplitInfo, leaf_objective, leaf_weight, split_gain,
};
pub use trainer::{BaseScore, GBTreeTrainer, QuantizedEvalSet, TrainerParams};
pub use tree::{
    BuildingNode, BuildingTree, DepthWisePolicy, DepthWiseState, GrowthPolicy, GrowthState,
    GrowthStrategy, LeafWisePolicy, LeafWiseState, NodeCandidate, TreeGrower, TreeParams,
};
