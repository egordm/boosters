//! GBLinear (linear booster) inference.
//!
//! This module provides linear model storage and prediction:
//!
//! - [`LinearModel`]: Weight matrix storage with feature-major layout
//!
//! # Linear Model
//!
//! Linear boosting uses weighted sums of features instead of decision trees:
//!
//! ```text
//! output[g] = base_score + bias[g] + Σ(feature[i] × weight[i, g])
//! ```
//!
//! The weight matrix is stored in feature-major, group-minor order with bias
//! in the last row:
//!
//! ```text
//! weights[feature * num_groups + group] → coefficient
//! weights[num_features * num_groups + group] → bias
//! ```

mod model;

pub use model::LinearModel;
