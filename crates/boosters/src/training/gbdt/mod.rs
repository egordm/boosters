//! Gradient Boosted Decision Tree (GBDT) training module.
//!
//! This module contains all components specific to tree-based gradient boosting:
//!
//! - [`categorical`] - Categorical feature utilities (CatBitset)
//! - [`expansion`] - Expansion strategies (depth-wise, leaf-wise)
//! - [`grower`] - Main tree growing orchestration
//! - [`histograms`] - Histogram data structures for gradient accumulation
//! - [`linear`] - Linear leaf training components
//! - [`parallelism`] - Parallelism configuration with self-correction
//! - [`partition`] - Row index partitioning for tree nodes
//! - [`split`] - Split types, gain computation, and finding algorithms
//! - [`trainer`] - GBDT training loop

pub mod categorical;
pub mod expansion;
pub mod grower;
pub mod histograms;
pub mod linear;
pub mod partition;
pub mod split;
pub mod trainer;

// Re-export main types
pub use categorical::CatBitset;
pub use expansion::{GrowthState, GrowthStrategy, NodeCandidate};
pub use grower::{GrowerParams, TreeGrower};
pub use histograms::{
    FeatureView, HistogramBin, HistogramBuilder, HistogramLayout, HistogramPool, HistogramSlot,
    HistogramSlotMut,
};
pub use linear::{
    LeafFeatureBuffer, LinearFeatureSelection, LinearLeafConfig, WeightedLeastSquaresSolver,
};
pub use partition::{LeafId, RowPartitioner};
pub use split::{DEFAULT_MAX_ONEHOT_CATS, GainParams, GreedySplitter, SplitInfo, SplitType};
pub use trainer::{GBDTParams, GBDTTrainer};
