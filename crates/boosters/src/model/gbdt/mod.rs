//! GBDT model and configuration.
//!
//! This module provides the high-level [`GBDTModel`] wrapper and nested
//! parameter structs for configuration.
//!
//! # Parameter Groups
//!
//! Configuration is organized into semantic groups:
//! - [`TreeParams`]: Tree structure (growth strategy, max depth/leaves)
//! - [`RegularizationParams`]: L1/L2 regularization and split constraints
//! - [`SamplingParams`]: Row and column subsampling
//!
//! These are used by both high-level [`GBDTConfig`] (if defined) and the
//! mid-level trainer parameters.

mod model;
mod params;

pub use model::GBDTModel;
pub use params::{
    ParamValidationError, RegularizationParams, SamplingParams, TreeParams,
};
