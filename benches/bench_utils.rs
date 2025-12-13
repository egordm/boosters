//! Shared utilities for benchmarks.
//!
//! This module provides common setup code used across multiple benchmarks.

#![allow(dead_code)]

use rand::prelude::*;
use std::fs::File;
use std::path::PathBuf;

use booste_rs::compat::XgbModel;
use booste_rs::inference::gbdt::{Forest, ScalarLeaf};
use booste_rs::data::RowMatrix;

/// Minimal model wrapper for benchmarks.
///
/// Benchmarks typically need only the parsed tree ensemble plus metadata like feature count.
pub struct LoadedForestModel {
    pub forest: Forest<ScalarLeaf>,
    pub num_features: usize,
}

// =============================================================================
// Data Generation
// =============================================================================

/// Generate random dense input data for prediction benchmarks.
pub fn generate_random_input(num_rows: usize, num_features: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..num_rows * num_features)
        .map(|_| rng.r#gen::<f32>() * 10.0 - 5.0) // Range [-5, 5]
        .collect()
}

/// Generate random dense training data.
///
/// Returns (features, labels) where labels are a simple linear function of features.
pub fn generate_training_data(
    num_rows: usize,
    num_features: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate random features
    let features: Vec<f32> = (0..num_rows * num_features)
        .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0) // Range [-1, 1]
        .collect();

    // Generate random true weights
    let true_weights: Vec<f32> = (0..num_features)
        .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
        .collect();
    let true_bias: f32 = rng.r#gen::<f32>() * 0.5;

    // Generate labels: y = X @ w + bias + noise
    let labels: Vec<f32> = (0..num_rows)
        .map(|row| {
            let row_start = row * num_features;
            let mut y = true_bias;
            for (j, &w) in true_weights.iter().enumerate() {
                y += features[row_start + j] * w;
            }
            y += rng.r#gen::<f32>() * 0.1 - 0.05; // Small noise
            y
        })
        .collect();

    (features, labels)
}

// =============================================================================
// Model Loading
// =============================================================================

/// Path to benchmark models directory.
pub fn bench_models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test-cases/benchmark")
}

/// Load a booste-rs model from JSON file.
pub fn load_boosters_model(name: &str) -> LoadedForestModel {
    let path = bench_models_dir().join(format!("{}.model.json", name));
    let file = File::open(&path).unwrap_or_else(|_| panic!("Failed to open model: {:?}", path));
    let xgb_model: XgbModel = serde_json::from_reader(file).expect("Failed to parse model");

    let forest = xgb_model.to_forest().expect("Failed to convert model to Forest");
    let num_features = xgb_model.learner.learner_model_param.num_feature as usize;

    LoadedForestModel { forest, num_features }
}

/// Create a RowMatrix from generated data.
pub fn create_matrix(num_rows: usize, num_features: usize, seed: u64) -> RowMatrix<f32, Box<[f32]>> {
    let data = generate_random_input(num_rows, num_features, seed);
    RowMatrix::from_vec(data, num_rows, num_features)
}
