//! Integration tests grouped by compatibility layer.
//!
//! This crate groups the feature-gated compat test suites so the `tests/` root
//! stays tidy.

#[cfg(feature = "xgboost-compat")]
#[path = "compat/test_data.rs"]
mod test_data;

#[cfg(feature = "lightgbm-compat")]
#[path = "compat/lightgbm.rs"]
mod lightgbm;

#[cfg(feature = "xgboost-compat")]
#[path = "compat/xgboost.rs"]
mod xgboost;
