//! GBLinear model and configuration.
//!
//! This module provides the high-level [`GBLinearModel`] wrapper and
//! configuration types for linear gradient boosting.
//!
//! # Example
//!
//! ```
//! use boosters::model::gblinear::GBLinearConfig;
//!
//! // Default config: regression with squared loss
//! let config = GBLinearConfig::builder().build().unwrap();
//!
//! // Classification with L1 regularization
//! use boosters::training::Objective;
//! let config = GBLinearConfig::builder()
//!     .objective(Objective::LogisticLoss)
//!     .n_rounds(200)
//!     .alpha(0.1)
//!     .build()
//!     .unwrap();
//! ```

mod config;
mod model;

pub use config::{ConfigError, GBLinearConfig, RegularizationParams};
pub use model::GBLinearModel;
