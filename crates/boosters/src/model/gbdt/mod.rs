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
//! These are composed into [`GBDTConfig`] for high-level model training.
//!
//! # Example
//!
//! ```
//! use boosters::model::gbdt::{GBDTConfig, TreeParams, SamplingParams};
//!
//! // Build config with custom settings (validation at build time)
//! let config = GBDTConfig::builder()
//!     .n_trees(200)
//!     .learning_rate(0.1)
//!     .tree(TreeParams::depth_wise(8))
//!     .sampling(SamplingParams { subsample: 0.8, ..Default::default() })
//!     .build()
//!     .unwrap();
//! ```

mod config;
mod model;
mod params;

pub use config::{ConfigError, GBDTConfig};
pub use model::GBDTModel;
pub use params::{
    ParamValidationError, RegularizationParams, SamplingParams, TreeParams,
};
