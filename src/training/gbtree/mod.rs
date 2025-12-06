//! GBTree (decision tree) training infrastructure.
//!
//! This module provides histogram-based gradient boosting for tree models,
//! following XGBoost and LightGBM approaches.
//!
//! # Overview
//!
//! Training workflow:
//! 1. Quantize features into bins using [`Quantizer`]
//! 2. Build trees using [`GBTreeTrainer`] with a [`GrowthPolicy`]
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
//! - [`GrowthPolicy`]: Strategy for expanding nodes
//! - [`DepthWisePolicy`]: XGBoost-style level-by-level growth
//! - [`LeafWisePolicy`]: LightGBM-style best-leaf-first growth
//!
//! # Histogram Building
//!
//! - [`HistogramBuilder`]: Accumulates gradient/hessian histograms
//! - [`NodeHistogram`]: Per-feature histograms for a node
//! - [`HistogramSubtractor`]: Parent-child histogram trick
//!
//! See RFC-0015 for tree growing design, RFC-0011 for quantization.

mod histogram;
mod partition;
mod quantize;
mod split;
mod trainer;
mod tree;

pub use histogram::{
    ChildSide, FeatureHistogram, HistogramBuilder, HistogramSubtractor, NodeHistogram,
};
pub use partition::RowPartitioner;
pub use quantize::{BinCuts, BinIndex, CutFinder, ExactQuantileCuts, QuantizedMatrix, Quantizer};
pub use split::{GainParams, GreedySplitFinder, SplitFinder, SplitInfo, leaf_objective, leaf_weight, split_gain};
pub use trainer::{BaseScore, GBTreeTrainer, QuantizedEvalSet, TrainerParams};
pub use tree::{
    BuildingNode, BuildingTree, DepthWisePolicy, DepthWiseState, GrowthPolicy, GrowthState,
    GrowthStrategy, LeafWisePolicy, LeafWiseState, NodeCandidate, TreeGrower, TreeParams,
};
