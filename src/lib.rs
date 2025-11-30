//! booste-rs: A gradient boosting library for Rust.
//!
//! This crate provides native Rust implementations for gradient boosted decision trees,
//! with support for loading models from external frameworks like XGBoost.

// Re-export approx traits for users who want to compare predictions
pub use approx;

pub mod compat;
pub mod data;
pub mod forest;
pub mod linear;
pub mod model;
pub mod objective;
pub mod predict;
pub mod training;
pub mod trees;
