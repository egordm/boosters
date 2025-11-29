//! Tree data structures.

pub mod array_layout;
pub mod categories;
pub mod leaf;
pub mod node;
pub mod soa;

pub use array_layout::{ArrayTreeLayout, MAX_UNROLL_DEPTH};
pub use categories::{categories_to_bitset, float_to_category, CategoriesStorage};
pub use leaf::{LeafValue, ScalarLeaf};
pub use node::{Node, SplitCondition};
pub use soa::{SoATreeStorage, TreeBuilder};
