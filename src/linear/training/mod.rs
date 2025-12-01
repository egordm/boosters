//! Linear model training via coordinate descent.
//!
//! This module provides training for linear models using coordinate descent
//! optimization with elastic net regularization (L1 + L2).
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

pub use selector::{CyclicSelector, FeatureSelector, ShuffleSelector};
pub use trainer::{LinearTrainer, LinearTrainerConfig};
pub use updater::{update_bias, UpdateConfig, UpdaterKind};
