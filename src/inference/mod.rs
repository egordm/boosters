//! Inference infrastructure for trained gradient boosting models.
//!
//! This module provides the prediction pipeline for both tree-based (GBDT) and
//! linear (GBLinear) models.
//!
//! # Module Structure
//!
//! - [`common`]: Shared types (`PredictionOutput`, output transforms)
//! - [`gbdt`]: Tree ensemble inference (`Forest`, `Tree`, predictors)
//! - [`gblinear`]: Linear model inference
//!
//! # Quick Start
//!
//! ```ignore
//! use booste_rs::inference::{Forest, Predictor, PredictionOutput};
//! use booste_rs::data::RowMatrix;
//!
//! // Load or build a forest
//! let forest: Forest = /* ... */;
//!
//! // Create predictor (optional optimization)
//! let predictor = Predictor::new(&forest);
//!
//! // Predict
//! let output = predictor.predict(&features);
//! ```

pub mod common;
pub mod gbdt;
pub mod gblinear;

// Re-export commonly used types
pub use common::PredictionOutput;
pub use gbdt::{
    Forest, Tree, MutableTree,
    Node, SplitCondition, SplitType,
    LeafValue, ScalarLeaf, VectorLeaf,
    CategoriesStorage,
    Predictor, SimplePredictor, UnrolledPredictor6,
    StandardTraversal, UnrolledTraversal, TreeTraversal,
    // Accessor utilities for generic traversal
    BinnedAccessor, traverse_to_leaf,
};
pub use gblinear::LinearModelPredict;
