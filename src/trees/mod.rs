//! Tree data structures.

pub mod leaf;
pub mod node;
pub mod soa;

pub use leaf::{LeafValue, ScalarLeaf};
pub use node::{Node, SplitCondition};
pub use soa::{SoATreeStorage, TreeBuilder};
