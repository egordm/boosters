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

pub use categories::{CategoriesStorage, categories_to_bitset, float_to_category};
pub use coefficients::{LeafCoefficients, LeafCoefficientsBuilder};
pub use forest::{Forest, ForestValidationError};
pub use mutable_tree::MutableTree;
pub use tree::Tree;
pub use tree_view::{TreeValidationError, TreeView, validate_tree};
pub use types::{LeafValue, ScalarLeaf, SplitType, VectorLeaf};
