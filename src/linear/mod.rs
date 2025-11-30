//! Linear booster (GBLinear) for gradient boosting.
//!
//! This module implements XGBoost's linear booster, which uses weighted sums
//! of features instead of decision trees. Prediction is a simple dot product:
//!
//! ```text
//! output[g] = base_score + bias[g] + Σ(feature[i] × weight[i, g])
//! ```
//!
//! See RFC-0008 for design rationale.

mod model;

pub use model::LinearModel;
