//! Explainability module.
//!
//! Provides feature importance and SHAP value computation.
//!
//! # Feature Importance
//!
//! Multiple importance types are supported:
//! - **Split**: Number of times each feature is used in splits
//! - **Gain**: Total gain from splits using each feature
//! - **AverageGain**: Gain divided by split count
//! - **Cover**: Total cover (sample weight) at nodes using each feature
//! - **AverageCover**: Cover divided by split count
//!
//! # Example
//!
//! ```ignore
//! use boosters::explainability::{ImportanceType, FeatureImportance};
//!
//! let importance = model.feature_importance(ImportanceType::Gain)?;
//! let top5 = importance.top_k(5);
//! ```

mod importance;

pub use importance::{
    compute_forest_importance, ExplainError, FeatureImportance, ImportanceType,
};
