//! Gradient-boosted decision tree (GBDT) canonical representations.

/// Canonical node identifier used by the GBDT representation.
///
/// Internally this is just an index into the tree's SoA arrays.
pub type NodeId = u32;

pub mod categories;
pub mod coefficients;
pub mod forest;
pub mod mutable_tree;
pub mod tree;
pub mod tree_view;
pub mod types;

pub use categories::{categories_to_bitset, float_to_category, CategoriesStorage};
pub use coefficients::{LeafCoefficients, LeafCoefficientsBuilder};
pub use forest::{Forest, ForestValidationError};
pub use mutable_tree::MutableTree;
pub use tree::Tree;
pub use tree_view::{validate_tree, TreeValidationError, TreeView};
pub use types::{LeafValue, ScalarLeaf, SplitType, VectorLeaf};
