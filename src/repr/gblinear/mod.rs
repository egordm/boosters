//! Gradient-boosted linear (GBLinear) canonical representations.
//!
//! This module defines the core linear model representation used by the
//! linear booster. The [`LinearModel`] type stores a weight matrix in
//! feature-major layout for efficient inference and training.
//!
//! # Weight Layout
//!
//! The weight matrix is stored as a flat array in feature-major, group-minor order:
//!
//! ```text
//! weights[feature * num_groups + group] → coefficient
//! weights[num_features * num_groups + group] → bias
//! ```
//!
//! # Example
//!
//! ```
//! use booste_rs::repr::gblinear::LinearModel;
//!
//! // Create a simple linear model: y = 0.5*x0 + 0.3*x1 + 0.1
//! let weights = vec![0.5, 0.3, 0.1].into_boxed_slice();
//! let model = LinearModel::new(weights, 2, 1);
//!
//! assert_eq!(model.weight(0, 0), 0.5);
//! assert_eq!(model.weight(1, 0), 0.3);
//! assert_eq!(model.bias(0), 0.1);
//! ```

mod model;

pub use model::LinearModel;
