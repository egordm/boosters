//! Tree growing infrastructure.
//!
//! This module coordinates tree building during gradient boosting training:
//!
//! - [`BuildingTree`]: Mutable tree structure during construction
//! - [`TreeGrower`]: Orchestrates histogram building, split finding, and node expansion
//! - [`GrowthStrategy`]: Enum for selecting depth-wise or leaf-wise growth
//!
//! # Growth Strategies
//!
//! - **Depth-wise**: XGBoost style, expand all nodes at each level
//! - **Leaf-wise**: LightGBM style, always expand best-gain leaf
//!
//! # Example
//!
//! ```ignore
//! use booste_rs::training::{TreeGrower, GrowthStrategy, TreeBuildParams};
//!
//! let strategy = GrowthStrategy::DepthWise { max_depth: 6 };
//! let params = TreeBuildParams::default();
//! let mut grower = TreeGrower::new(strategy, &cuts, params, learning_rate, ...);
//!
//! let tree = grower.build_tree(&quantized, &grads, &mut partitioner, seed);
//! ```
//!
//! See RFC-0015 for design rationale.

mod building;
mod grower;
mod policy;

pub use building::{BuildingNode, BuildingTree, NodeCandidate};
pub use grower::{ParallelStrategy, TreeBuildParams, TreeGrower};
pub use policy::GrowthStrategy;
