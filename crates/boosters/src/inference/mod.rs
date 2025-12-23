//! Inference infrastructure for trained gradient boosting models.
//!
//! This module provides the prediction pipeline for tree-based (GBDT) models.
//! For linear model prediction, see [`repr::gblinear::LinearModel`](crate::repr::gblinear::LinearModel).
//!
//! # Module Structure
//!
//! - [`predictions`]: Semantic prediction wrappers (`PredictionKind`, `Predictions`)
//! - [`gbdt`]: Tree ensemble inference (predictors, traversal strategies)
//!
//! # Quick Start
//!
//! ```ignore
//! use boosters::repr::gbdt::Forest;
//! use boosters::inference::gbdt::{Predictor, UnrolledTraversal6};
//!
//! // Load or build a forest
//! let forest: Forest = /* ... */;
//!
//! // Create predictor with traversal strategy
//! let predictor = Predictor::<UnrolledTraversal6>::new(&forest);
//!
//! // Predict - returns Array2<f32> with shape (n_samples, n_groups)
//! let output = predictor.predict(&features);
//! ```

mod predictions;
pub mod gbdt;

// Re-export commonly used inference types
pub use predictions::{PredictionKind, Predictions};
pub use gbdt::{
    Predictor, SimplePredictor, UnrolledPredictor6,
    StandardTraversal, UnrolledTraversal, UnrolledTraversal6, TreeTraversal,
    BinnedAccessor,
};
