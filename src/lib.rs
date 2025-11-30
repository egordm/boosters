//! booste-rs: A gradient boosting library for Rust.
//!
//! This crate provides native Rust implementations for gradient boosted decision trees,
//! with support for loading models from external frameworks like XGBoost.

pub mod compat;
pub mod data;
pub mod forest;
pub mod linear;
pub mod model;
pub mod objective;
pub mod predict;
pub mod trees;
