//! Tree data structures.

pub mod leaf;
pub mod node;

pub use leaf::{LeafValue, ScalarLeaf};
pub use node::{Node, SplitCondition};
