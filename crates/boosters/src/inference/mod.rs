//! Inference infrastructure for trained gradient boosting models.
//!
//! This module provides the prediction pipeline for both tree-based (GBDT) and
//! linear (GBLinear) models.
//!
//! # Module Structure
//!
//! - [`common`]: Shared types (`PredictionOutput`, output transforms)
//! - [`gbdt`]: Tree ensemble inference (predictors, traversal strategies)
//! - [`gblinear`]: Linear model inference
//!
//! # Quick Start
//!
//! ```ignore
//! use boosters::repr::gbdt::Forest;
//! use boosters::inference::gbdt::{Predictor, UnrolledTraversal6};
//! use boosters::inference::PredictionOutput;
//!
//! // Load or build a forest
//! let forest: Forest = /* ... */;
//!
//! // Create predictor with traversal strategy
//! let predictor = Predictor::<UnrolledTraversal6>::new(&forest);
//!
//! // Predict
//! let output = predictor.predict(&features);
//! ```

pub mod common;
pub mod gbdt;
pub mod gblinear;

// Re-export commonly used inference types
pub use common::PredictionOutput;
pub use gbdt::{
    Predictor, SimplePredictor, UnrolledPredictor6,
    StandardTraversal, UnrolledTraversal, UnrolledTraversal6, TreeTraversal,
    BinnedAccessor, traverse_to_leaf,
};
pub use gblinear::LinearModelPredict;
