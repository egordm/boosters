//! Sampling strategies for gradient boosting.
//!
//! This module provides row and column sampling strategies used during training
//! for regularization and computational efficiency.
//!
//! # Row Sampling
//!
//! - **Uniform**: Standard bagging (random subsampling)
//! - **GOSS**: Gradient-based One-Side Sampling (keeps top gradients, samples rest)
//!
//! # Column Sampling
//!
//! - **bytree**: Sample features once per tree
//! - **bylevel**: Sample features at each depth level
//! - **bynode**: Sample features at each node

pub mod column;
pub mod row;

pub use column::{ColSampler, ColSamplingParams};
pub use row::{RowSampler, RowSamplingParams};
