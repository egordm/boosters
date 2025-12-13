//! Tree module for gradient boosting.
//!
//! This module provides tree data structures:
//! - [`Tree`], [`TreeNode`]: Immutable tree structures for prediction
//! - [`TreeBuilder`]: Mutable builder for tree construction
//! - [`Forest`]: Collection of trees (ensemble)
//! - [`NodeId`]: Type alias for tree node indices

pub mod builder;
pub mod forest;
pub mod node;

pub use builder::TreeBuilder;
pub use forest::Forest;
pub use node::{CategoricalSplit, NodeId, Tree, TreeNode, NO_CHILD};
