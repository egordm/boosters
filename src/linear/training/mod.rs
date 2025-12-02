//! Linear model training via coordinate descent.
//!
//! This module provides training for linear models using coordinate descent
//! optimization with elastic net regularization (L1 + L2).
//!
//! # Feature Selectors
//!
//! The order of feature updates affects convergence. Available selectors:
//!
//! | Selector | XGBoost | Description |
//! |----------|---------|-------------|
//! | [`CyclicSelector`] | `cyclic` | Sequential order (0, 1, 2, ...) |
//! | [`ShuffleSelector`] | `shuffle` | Random permutation each round |
//! | [`RandomSelector`] | `random` | Random with replacement |
//! | [`GreedySelector`] | `greedy` | Largest gradient magnitude first |
//! | [`ThriftySelector`] | `thrifty` | Approximate greedy (sort once) |
//!
//! Use [`FeatureSelectorKind`] to configure the selector in [`LinearTrainerConfig`].
//!
//! # Gradient Storage
//!
//! Gradients are stored in Structure-of-Arrays (SoA) layout via [`GradientBuffer`]:
//! - Shape `[n_samples, n_outputs]` for unified single/multi-output handling
//! - Separate `grads[]` and `hess[]` arrays for cache efficiency
//!
//! See RFC-0009 for design rationale.

mod selector;
mod trainer;
mod updater;

pub use selector::{
    CyclicSelector, FeatureSelector, FeatureSelectorKind, GreedySelector, RandomSelector,
    SelectorState, ShuffleSelector, ThriftySelector,
};
pub use trainer::{LinearTrainer, LinearTrainerConfig};
pub use updater::{update_bias, UpdateConfig, UpdaterKind};
