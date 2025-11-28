//! Tree data structures.

pub mod categories;
pub mod leaf;
pub mod node;
pub mod soa;

pub use categories::{categories_to_bitset, float_to_category, CategoriesStorage};
pub use leaf::{LeafValue, ScalarLeaf};
pub use node::{Node, SplitCondition};
pub use soa::{SoATreeStorage, TreeBuilder};
