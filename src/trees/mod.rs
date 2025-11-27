//! Tree data structures.

pub mod leaf;
pub mod node;
pub mod storage;

pub use leaf::{LeafValue, ScalarLeaf};
pub use node::{Node, SplitCondition};
pub use storage::{SoATreeStorage, TreeBuilder};
