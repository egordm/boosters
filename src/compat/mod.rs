//! External format compatibility loaders.
//!
//! This module provides loaders for models trained in external frameworks
//! (XGBoost, LightGBM, etc.) and converts them to native booste-rs types.
//!
//! Each loader is feature-gated to avoid pulling in unnecessary dependencies.

#[cfg(feature = "xgboost-compat")]
pub mod xgboost;

#[cfg(feature = "xgboost-compat")]
pub use xgboost::{ConversionError, XgbModel};
