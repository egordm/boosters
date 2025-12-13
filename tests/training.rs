//! Integration tests grouped by training subsystem.

#[path = "training/gbdt.rs"]
mod gbdt;

// GBLinear integration tests.
#[path = "training/gblinear/mod.rs"]
mod gblinear;

