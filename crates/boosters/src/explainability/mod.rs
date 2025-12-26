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
//! # SHAP Values
//!
//! SHAP (SHapley Additive exPlanations) values explain individual predictions:
//! - **TreeExplainer**: For tree ensembles (GBDT, Random Forest)
//! - **LinearExplainer**: For linear models (closed-form solution)
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
pub mod shap;

pub use importance::{ExplainError, FeatureImportance, ImportanceType, compute_forest_importance};
pub use shap::{LinearExplainer, PathState, ShapValues, TreeExplainer};
