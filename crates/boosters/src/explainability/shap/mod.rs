//! SHAP (SHapley Additive exPlanations) value computation.
//!
//! This module provides TreeSHAP and Linear SHAP implementations for
//! model explainability.

mod linear_explainer;
mod path;
mod tree_explainer;
mod values;

pub use linear_explainer::LinearExplainer;
pub use path::PathState;
pub use tree_explainer::TreeExplainer;
pub use values::ShapValues;
