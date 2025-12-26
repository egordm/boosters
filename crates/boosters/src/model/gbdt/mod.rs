//! GBDT model and configuration.
//!
//! This module provides the high-level [`GBDTModel`] wrapper and configuration
//! for GBDT training.
//!
//! # Example
//!
//! ```
//! use boosters::model::gbdt::GBDTConfig;
//!
//! // Build config with custom settings (validation at build time)
//! use boosters::training::GrowthStrategy;
//! let config = GBDTConfig::builder()
//!     .n_trees(200)
//!     .learning_rate(0.1)
//!     .growth_strategy(GrowthStrategy::DepthWise { max_depth: 8 })
//!     .subsample(0.8)
//!     .build()
//!     .unwrap();
//! ```

mod config;
mod model;

pub use config::{ConfigError, GBDTConfig};
pub use model::GBDTModel;
