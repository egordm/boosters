//! Gradient Boosted Decision Tree (GBDT) training module.
//!
//! This module contains all components specific to tree-based gradient boosting:
//!
//! - [`categorical`] - Categorical feature utilities (CatBitset)
//! - [`expansion`] - Expansion strategies (depth-wise, leaf-wise)
//! - [`grower`] - Main tree growing orchestration
//! - [`histograms`] - Histogram data structures for gradient accumulation
//! - [`optimization`] - Auto-selection of optimization strategies
//! - [`partition`] - Row index partitioning for tree nodes
//! - [`split`] - Split types, gain computation, and finding algorithms
//! - [`trainer`] - GBDT training loop
//! - [`tree`] - Tree structures and construction

pub mod categorical;
pub mod expansion;
pub mod grower;
pub mod histograms;
pub mod optimization;
pub mod partition;
pub mod split;
pub mod trainer;
pub mod tree;

mod convert;

// Re-export main types
pub use categorical::CatBitset;
pub use expansion::{GrowthState, GrowthStrategy, NodeCandidate};
pub use grower::{TreeGrower, GrowerParams};
pub use histograms::{
    FeatureMeta, FeatureView, HistogramBin, HistogramPool, HistogramSlot,
    HistogramSlotMut, ParallelStrategy as HistogramParallelStrategy,
};
pub use crate::training::Gradients;
pub use optimization::OptimizationProfile;
pub use partition::{LeafId, RowPartitioner};
pub use split::{GainParams, GreedySplitter, SplitInfo, SplitStrategy, SplitType, DEFAULT_MAX_ONEHOT_CATS};
pub use trainer::{GBDTParams, GBDTTrainer};
pub use tree::{Forest, NodeId, Tree, TreeBuilder, TreeNode};
