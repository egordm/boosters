//! Tree data structures.

pub mod categories;
pub mod leaf;
pub mod node;
pub mod soa;
pub mod unrolled_layout;

pub use categories::{categories_to_bitset, float_to_category, CategoriesStorage};
pub use leaf::{LeafValue, ScalarLeaf};
pub use node::{Node, SplitCondition};
pub use soa::{SoATreeStorage, TreeBuilder};
pub use unrolled_layout::{
    nodes_at_depth, Depth4, Depth6, Depth8, UnrollDepth, UnrolledTreeLayout, UnrolledTreeLayout4,
    UnrolledTreeLayout6, UnrolledTreeLayout8, DEFAULT_UNROLL_DEPTH, MAX_UNROLL_DEPTH,
};
