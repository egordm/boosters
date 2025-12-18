//! Gradient-boosted decision tree (GBDT) canonical representations.

/// Canonical node identifier used by the GBDT representation.
///
/// Internally this is just an index into the tree's SoA arrays.
pub type NodeId = u32;

pub mod categories;
pub mod coefficients;
pub mod forest;
pub mod leaf;
pub mod node;
pub mod tree;

pub use categories::{categories_to_bitset, float_to_category, CategoriesStorage};
pub use coefficients::{LeafCoefficients, LeafCoefficientsBuilder};
pub use forest::{Forest, ForestValidationError};
pub use leaf::{LeafValue, ScalarLeaf, VectorLeaf};
pub use node::{Node, SplitCondition, SplitType};
pub use tree::{MutableTree, Tree, TreeValidationError, TreeView};
