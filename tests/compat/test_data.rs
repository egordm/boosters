//! Test data loading utilities for compatibility test cases.
//!
//! Currently used by the XGBoost compat integration tests.

#![allow(dead_code)]

use std::fs::File;
use std::path::PathBuf;

use serde::de::DeserializeOwned;

// Re-export test data structures from library
pub use boosters::testing::{TestExpected, TestInput, DEFAULT_TOLERANCE_F64};

/// Base directory for test cases.
pub fn test_cases_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test-cases")
}

/// Directory for XGBoost test cases.
pub fn xgboost_test_cases_dir() -> PathBuf {
    test_cases_dir().join("xgboost")
}

/// Load a JSON file and deserialize it.
pub fn load_json<T: DeserializeOwned>(path: &std::path::Path) -> T {
    let file =
        File::open(path).unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
    serde_json::from_reader(file)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()))
}

/// A complete test case with model, input, and expected output.
pub struct TestCase<M> {
    pub name: String,
    pub model: M,
    pub input: TestInput,
    pub expected: TestExpected,
}
