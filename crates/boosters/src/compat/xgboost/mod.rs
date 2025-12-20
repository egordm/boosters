//! XGBoost JSON model format support.
//!
//! This module provides parsing of XGBoost's JSON model format and conversion
//! to native booste-rs types.

mod convert;
mod json;

pub use convert::{Booster, ConversionError};
pub use json::*;
