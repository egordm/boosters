//! Linear leaf training components.
//!
//! This module provides support for fitting linear models at tree leaves,
//! enabling smoother predictions within each leaf partition.
//!
//! # Components
//!
//! - [`LeafFeatureBuffer`]: Column-major buffer for gathering leaf features
//! - [`WeightedLeastSquaresSolver`]: Coordinate descent solver for weighted least squares
//! - [`LinearLeafConfig`]: Configuration for linear leaf training
//! - [`LeafLinearTrainer`]: Orchestrates linear fitting for all leaves
//!
//! # Design
//!
//! Linear leaves fit `intercept + Σ(coef × feature)` at each leaf.
//! See RFC-0010 for design rationale.

mod buffer;
mod config;
mod solver;
mod trainer;

pub use buffer::LeafFeatureBuffer;
pub use config::{LinearFeatureSelection, LinearLeafConfig};
pub use solver::WeightedLeastSquaresSolver;
pub use trainer::{FittedLeaf, LeafLinearTrainer};
