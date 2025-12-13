//! GBDT (Gradient Boosted Decision Trees) inference.
//!
//! This module provides tree ensemble storage and prediction:
//!
//! - [`Tree`]: Structure-of-Arrays storage for a single tree
//! - [`Forest`]: Collection of trees with group assignments
//! - [`Predictor`]: High-performance batch prediction
//!
//! # Tree Storage
//!
//! Trees are stored in Structure-of-Arrays (SoA) format for cache-efficient
//! traversal. Each tree's nodes are stored in parallel arrays:
//!
//! - `split_indices`: Feature index to split on
//! - `split_thresholds`: Threshold for numeric splits
//! - `left_children`, `right_children`: Child node indices
//! - `default_left`: Direction for missing values
//! - `is_leaf`: Whether node is a leaf
//! - `leaf_values`: Prediction values at leaves
//!
//! # Categorical Splits
//!
//! Categorical splits use bitset storage. A category goes RIGHT if its bit
//! is set in the bitset, LEFT otherwise.
//!
//! # Predictor Optimization
//!
//! The [`Predictor`] provides block-based and unrolled traversal optimizations:
//!
//! - [`StandardTraversal`]: Simple per-node traversal
//! - [`UnrolledTraversal`]: Pre-computes flat layout for top tree levels
//!
//! For large batches (100+ rows), unrolled traversal is 2-3x faster.

mod predictor;
mod traversal;
mod unrolled;

#[cfg(feature = "simd")]
mod simd;

// Re-export canonical representation types from `repr`.
pub use crate::repr::gbdt::{
    categories_to_bitset, float_to_category,
    CategoriesStorage, Forest, LeafValue, MutableTree, Node, ScalarLeaf, SplitCondition,
    SplitType, Tree, VectorLeaf,
};
pub use unrolled::{
    UnrolledTreeLayout, UnrolledTreeLayout4, UnrolledTreeLayout6, UnrolledTreeLayout8,
    UnrollDepth, Depth4, Depth6, Depth8,
    nodes_at_depth, DEFAULT_UNROLL_DEPTH, MAX_UNROLL_DEPTH,
};

// Re-export predictor types
pub use predictor::{
    Predictor, SimplePredictor, UnrolledPredictor4, UnrolledPredictor6, UnrolledPredictor8,
    DEFAULT_BLOCK_SIZE,
};

// Re-export traversal types
pub use traversal::{
    StandardTraversal, TreeTraversal, UnrolledTraversal, UnrolledTraversal4, UnrolledTraversal6,
    UnrolledTraversal8, traverse_from_node,
};

// Re-export SIMD types when feature is enabled
#[cfg(feature = "simd")]
pub use simd::{SimdTraversal, SimdTraversal4, SimdTraversal6, SimdTraversal8, SIMD_WIDTH};

#[cfg(feature = "simd")]
pub type SimdPredictor4<'f> = Predictor<'f, SimdTraversal4>;
#[cfg(feature = "simd")]
pub type SimdPredictor6<'f> = Predictor<'f, SimdTraversal6>;
#[cfg(feature = "simd")]
pub type SimdPredictor8<'f> = Predictor<'f, SimdTraversal8>;
