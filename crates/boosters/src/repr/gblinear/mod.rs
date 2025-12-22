//! Gradient-boosted linear (GBLinear) canonical representations.
//!
//! This module defines the core linear model representation used by the
//! linear booster. The [`LinearModel`] type stores a weight matrix using
//! ndarray for efficient inference and training.
//!
//! # Weight Layout
//!
//! The weight matrix has shape `[n_features + 1, n_groups]`:
//!
//! ```text
//! weights[[feature, group]] → coefficient
//! weights[[n_features, group]] → bias (last row)
//! ```
//!
//! # Example
//!
//! ```
//! use boosters::repr::gblinear::LinearModel;
//! use ndarray::array;
//!
//! // Create a simple linear model: y = 0.5*x0 + 0.3*x1 + 0.1
//! let weights = array![[0.5], [0.3], [0.1]];
//! let model = LinearModel::new(weights);
//!
//! assert_eq!(model.weight(0, 0), 0.5);
//! assert_eq!(model.weight(1, 0), 0.3);
//! assert_eq!(model.bias(0), 0.1);
//! ```

mod model;

pub use model::LinearModel;
