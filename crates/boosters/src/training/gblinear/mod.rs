//! GBLinear training via coordinate descent.
//!
//! This module provides training for linear models using coordinate descent
//! optimization with elastic net regularization (L1 + L2).
//!
//! # Example
//!
//! ```ignore
//! use boosters::training::{GBLinearParams, GBLinearTrainer, Metric, Objective};
//!
//! let params = GBLinearParams {
//!     n_rounds: 100,
//!     learning_rate: 0.5,
//!     lambda: 1.0,
//!     ..Default::default()
//! };
//!
//! let trainer = GBLinearTrainer::new(Objective::SquaredLoss, Metric::Rmse, params);
//! let model = trainer.train(&data, &labels, None, None).unwrap();
//! ```
//!
//! # Feature Selectors
//!
//! The order of feature updates affects convergence. Available selectors:
//!
//! | Selector | XGBoost | Description |
//! |----------|---------|-------------|
//! | Cyclic | `cyclic` | Sequential order (0, 1, 2, ...) |
//! | Shuffle | `shuffle` | Random permutation each round |
//! | Random | `random` | Random with replacement |
//! | Greedy | `greedy` | Largest gradient magnitude first |
//! | Thrifty | `thrifty` | Approximate greedy (sort once) |
//!
//! Use [`FeatureSelectorKind`] to configure the selector in [`GBLinearParams`].
//!
//! # Gradient Storage
//!
//! Gradients are stored in Structure-of-Arrays (SoA) layout via [`Gradients`][crate::training::Gradients]:
//! - Shape `[n_samples, n_outputs]` for unified single/multi-output handling
//! - Separate `grads[]` and `hess[]` arrays for cache efficiency
//!
//! See RFC-0009 for design rationale.

mod error;
mod selector;
mod trainer;
mod updater;

pub use error::GBLinearTrainError;
pub use selector::FeatureSelectorKind;
pub use trainer::{GBLinearParams, GBLinearTrainer};
pub use updater::UpdateStrategy;
