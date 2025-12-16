//! GBLinear (linear booster) inference.
//!
//! This module provides linear model prediction capabilities:
//!
//! - [`LinearModel`](crate::repr::gblinear::LinearModel): Weight matrix storage
//! - [`LinearModelPredict`]: Prediction trait for batch and single-row inference
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
//!
//! # Usage
//!
//! ```
//! use booste_rs::repr::gblinear::LinearModel;
//! use booste_rs::inference::gblinear::LinearModelPredict;
//!
//! let weights = vec![0.5, 0.3, 0.1].into_boxed_slice();
//! let model = LinearModel::new(weights, 2, 1);
//!
//! let output = model.predict_row(&[2.0, 3.0], &[0.0]);
//! ```

mod predict;
pub use predict::LinearModelPredict;
