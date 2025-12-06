//! Tree growing infrastructure.
//!
//! This module coordinates tree building during gradient boosting training:
//!
//! - [`BuildingTree`]: Mutable tree structure during construction
//! - [`TreeGrower`]: Orchestrates histogram building, split finding, and node expansion
//! - [`GrowthPolicy`]: Strategy for selecting nodes to expand
//!
//! # Growth Strategies
//!
//! - **Depth-wise** ([`DepthWisePolicy`]): XGBoost style, expand all nodes at each level
//! - **Leaf-wise** ([`LeafWisePolicy`]): LightGBM style, always expand best-gain leaf
//!
//! # Example
//!
//! ```ignore
//! use booste_rs::training::{TreeGrower, DepthWisePolicy, TreeParams};
//!
//! let policy = DepthWisePolicy { max_depth: 6 };
//! let params = TreeParams::default();
//! let mut grower = TreeGrower::new(policy, &cuts, params);
//!
//! let tree = grower.build_tree(&quantized, &grads, &mut partitioner);
//! ```
//!
//! See RFC-0015 for design rationale.

mod building;
mod grower;
mod policy;

pub use building::{BuildingNode, BuildingTree, NodeCandidate};
pub use grower::{TreeGrower, TreeParams};
pub use policy::{
    DepthWisePolicy, DepthWiseState, GrowthPolicy, GrowthState, GrowthStrategy, LeafCandidate,
    LeafWisePolicy, LeafWiseState,
};
