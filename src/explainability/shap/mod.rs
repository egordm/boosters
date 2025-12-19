//! SHAP (SHapley Additive exPlanations) value computation.
//!
//! This module provides TreeSHAP and Linear SHAP implementations for
//! model explainability.

mod path;
mod values;

pub use path::PathState;
pub use values::ShapValues;
