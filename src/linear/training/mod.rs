//! Linear model training via coordinate descent.
//!
//! This module provides training for linear models using coordinate descent
//! optimization with elastic net regularization (L1 + L2).
//!
//! See RFC-0009 for design rationale.

mod selector;
mod trainer;
mod updater;

pub use selector::{CyclicSelector, FeatureSelector, ShuffleSelector};
pub use trainer::{LinearTrainer, LinearTrainerConfig};
pub use updater::{CoordinateUpdater, ShotgunUpdater, Updater};
