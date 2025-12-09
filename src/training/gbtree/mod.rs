//! GBTree (decision tree) training infrastructure.
//!
//! This module provides histogram-based gradient boosting for tree models,
//! following XGBoost and LightGBM approaches.
//!
//! # Overview
//!
//! Training workflow:
//! 1. Quantize features into bins using [`Quantizer`]
//! 2. Build trees using [`GBTreeTrainer`] with a [`GrowthStrategy`]
//! 3. Convert to inference format ([`SoAForest`][crate::forest::SoAForest])
//!
//! # Quantization (RFC-0011)
//!
//! - [`BinCuts`]: Bin boundaries for all features
//! - [`QuantizedMatrix`]: Quantized feature storage
//! - [`Quantizer`]: Transforms raw features to bins
//!
//! # Tree Growing
//!
//! - [`TreeGrower`]: Builds a single tree from gradients
//! - [`GrowthStrategy`]: Enum for depth-wise or leaf-wise growth
//!
//! # Histogram Building (RFC-0025)
//!
//! - [`HistogramBuilder`]: Accumulates gradient/hessian histograms
//! - Pool-based storage with LRU cache for memory efficiency
//!
//! Histograms implement `Sub<&Self>` for efficient sibling derivation.
//!
//! See RFC-0015 for tree growing design, RFC-0011 for quantization.

mod grower;
mod histogram;
mod partition;
mod quantize;
mod sampling;
mod split;
mod trainer;

pub use grower::{
    BuildingNode, BuildingTree, GrowthStrategy, NodeCandidate, TreeBuildParams, TreeGrower,
};
pub use histogram::HistogramBuilder;
pub use partition::RowPartitioner;
pub use quantize::{
    BinCuts, BinIndex, CategoricalInfo, CutFinder, ExactQuantileCuts, QuantizedMatrix, Quantizer,
};
pub use sampling::{
    ColumnSampler, GossSampler, NoSampler, RandomSampler, RowSample, RowSampler, RowSampling,
};
pub use split::{GainParams, GreedySplitFinder, SplitFinder, SplitInfo, leaf_objective, leaf_weight, split_gain};
pub use trainer::{GBTreeTrainer, GBTreeTrainerBuilder, QuantizedEvalSet};
