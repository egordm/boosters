//! Synthetic test data generation utilities.
//!
//! This module provides utilities for generating synthetic datasets for testing
//! and benchmarking.
//!
//! # Preferred API
//!
//! Use the `synthetic_*` functions, which return a [`Dataset`] with targets set:
//!
//! ```
//! use boosters::data::{BinnedDataset, BinningConfig};
//! use boosters::testing::synthetic_datasets::synthetic_regression;
//!
//! let dataset = synthetic_regression(1000, 10, 42, 0.05);
//! let config = BinningConfig::builder().max_bins(256).build();
//! let binned = BinnedDataset::from_dataset(&dataset, &config).unwrap();
//! assert!(dataset.targets().is_some());
//! ```
//!
//! # Legacy API
//!
//! The `random_dense_f32` and `synthetic_*_targets_*` functions are deprecated.
//! They return raw `Vec<f32>` in row-major order requiring manual transposition.

use rand::prelude::*;

use ndarray::{Array1, Array2, ArrayView2};

use crate::data::{Dataset, Feature, transpose_to_c_order};

/// Get features in sample-major (row-major) layout.
///
/// Returns an Array2 with shape `[n_samples, n_features]`.
/// Useful for external libraries that expect row-major data.
pub fn features_row_major(dataset: &Dataset) -> Array2<f32> {
    let rows = dataset.n_samples();
    let cols = dataset.n_features();
    let data = features_row_major_slice(dataset);
    Array2::from_shape_vec((rows, cols), data).expect("shape mismatch")
}

/// Get features in sample-major (row-major) layout as a contiguous slice.
///
/// Useful for external libraries that need raw `&[f32]`.
pub fn features_row_major_slice(dataset: &Dataset) -> Vec<f32> {
    let rows = dataset.n_samples();
    let cols = dataset.n_features();

    let mut out = vec![0.0f32; rows * cols];

    for (feature_idx, feature) in dataset.feature_columns().iter().enumerate() {
        match feature {
            Feature::Dense(values) => {
                for sample_idx in 0..rows {
                    out[sample_idx * cols + feature_idx] = values[sample_idx];
                }
            }
            Feature::Sparse {
                indices,
                values,
                n_samples: _,
                default,
            } => {
                for sample_idx in 0..rows {
                    out[sample_idx * cols + feature_idx] = *default;
                }
                for (&row_idx, &val) in indices.iter().zip(values.iter()) {
                    let r = row_idx as usize;
                    out[r * cols + feature_idx] = val;
                }
            }
        }
    }

    out
}

/// Generate a synthetic regression dataset with linear targets.
///
/// This is the preferred API for benchmarks and tests. Returns feature-major
/// features ready for training, plus targets.
///
/// # Arguments
///
/// * `n_samples` - Number of samples
/// * `n_features` - Number of features
/// * `seed` - Random seed for reproducibility
/// * `noise` - Noise amplitude (0.0 = no noise)
///
/// # Example
///
/// ```
/// use boosters::data::{BinnedDataset, BinningConfig};
/// use boosters::testing::synthetic_datasets::synthetic_regression;
///
/// let dataset = synthetic_regression(1000, 10, 42, 0.05);
/// let config = BinningConfig::builder().max_bins(256).build();
/// let _binned = BinnedDataset::from_dataset(&dataset, &config).unwrap();
/// ```
pub fn synthetic_regression(n_samples: usize, n_features: usize, seed: u64, noise: f32) -> Dataset {
    let features_sm = random_features_array(n_samples, n_features, seed, -1.0, 1.0);
    let targets = generate_linear_targets(features_sm.view(), seed.wrapping_add(1), noise);
    let features_fm = transpose_to_c_order(features_sm.view());

    let targets_2d = targets.insert_axis(ndarray::Axis(0));
    Dataset::from_array(features_fm.view(), Some(targets_2d), None)
}

/// Generate a synthetic binary classification dataset.
///
/// Targets are 0.0 or 1.0 based on linear score thresholding.
pub fn synthetic_binary(n_samples: usize, n_features: usize, seed: u64, noise: f32) -> Dataset {
    let features_sm = random_features_array(n_samples, n_features, seed, -1.0, 1.0);
    let scores = generate_linear_targets(features_sm.view(), seed.wrapping_add(1), noise);
    let targets = scores.mapv(|s| if s > 0.0 { 1.0 } else { 0.0 });
    let features_fm = transpose_to_c_order(features_sm.view());

    let targets_2d = targets.insert_axis(ndarray::Axis(0));
    Dataset::from_array(features_fm.view(), Some(targets_2d), None)
}

/// Generate a synthetic multiclass classification dataset.
///
/// Targets are class indices (0, 1, 2, ..., n_classes-1) as f32.
pub fn synthetic_multiclass(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    seed: u64,
    noise: f32,
) -> Dataset {
    assert!(n_classes >= 2);
    let features_sm = random_features_array(n_samples, n_features, seed, -1.0, 1.0);
    let targets =
        generate_multiclass_targets(features_sm.view(), n_classes, seed.wrapping_add(1), noise);
    let features_fm = transpose_to_c_order(features_sm.view());

    let targets_2d = targets.insert_axis(ndarray::Axis(0));
    Dataset::from_array(features_fm.view(), Some(targets_2d), None)
}

// =============================================================================
// Internal Helpers
// =============================================================================

/// Generate linear regression targets from sample-major features.
fn generate_linear_targets(features: ArrayView2<'_, f32>, seed: u64, noise: f32) -> Array1<f32> {
    let (n_samples, n_features) = features.dim();
    let mut rng = StdRng::seed_from_u64(seed);

    // Random weights and bias
    let weights: Vec<f32> = (0..n_features)
        .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
        .collect();
    let bias: f32 = rng.r#gen::<f32>() * 0.5 - 0.25;

    let mut targets = Array1::zeros(n_samples);
    for r in 0..n_samples {
        let mut y = bias;
        for (c, &w) in weights.iter().enumerate() {
            y += features[[r, c]] * w;
        }
        if noise > 0.0 {
            y += (rng.r#gen::<f32>() * 2.0 - 1.0) * noise;
        }
        targets[r] = y;
    }

    targets
}

/// Generate multiclass targets from sample-major features.
fn generate_multiclass_targets(
    features: ArrayView2<'_, f32>,
    n_classes: usize,
    seed: u64,
    noise: f32,
) -> Array1<f32> {
    let (n_samples, n_features) = features.dim();
    let mut rng = StdRng::seed_from_u64(seed);

    // Random weights per class
    let weights: Vec<f32> = (0..n_classes * n_features)
        .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
        .collect();
    let bias: Vec<f32> = (0..n_classes)
        .map(|_| rng.r#gen::<f32>() * 0.5 - 0.25)
        .collect();

    let mut targets = Array1::zeros(n_samples);
    for r in 0..n_samples {
        let mut best_class = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (k, &b) in bias.iter().enumerate() {
            let mut s = b;
            let w_off = k * n_features;
            for c in 0..n_features {
                s += features[[r, c]] * weights[w_off + c];
            }
            if noise > 0.0 {
                s += (rng.r#gen::<f32>() * 2.0 - 1.0) * noise;
            }
            if s > best_score {
                best_score = s;
                best_class = k;
            }
        }
        targets[r] = best_class as f32;
    }

    targets
}

// =============================================================================
// Public Helpers
// =============================================================================

/// Create a sample-major [`Array2<f32>`] from random dense features.
///
/// Returns an array with shape `[rows, cols]` (sample-major).
pub fn random_features_array(
    rows: usize,
    cols: usize,
    seed: u64,
    min: f32,
    max: f32,
) -> Array2<f32> {
    assert!(max >= min);
    let mut rng = StdRng::seed_from_u64(seed);
    let width = max - min;
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| min + rng.r#gen::<f32>() * width)
        .collect();
    Array2::from_shape_vec((rows, cols), data).expect("shape mismatch")
}

/// Deterministic train/valid split indices.
///
/// Returns `(train_idx, valid_idx)`.
pub fn split_indices(rows: usize, valid_fraction: f32, seed: u64) -> (Vec<usize>, Vec<usize>) {
    assert!((0.0..1.0).contains(&valid_fraction));
    let mut idx: Vec<usize> = (0..rows).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    idx.shuffle(&mut rng);

    let valid_len = ((rows as f32) * valid_fraction).round() as usize;
    let valid_len = valid_len.min(rows);
    let (valid, train) = idx.split_at(valid_len);
    (train.to_vec(), valid.to_vec())
}

/// Select rows from a sample-major Array2 by indices.
///
/// Returns a new Array2 with shape `[indices.len(), n_features]`.
pub fn select_rows(features: ArrayView2<'_, f32>, indices: &[usize]) -> Array2<f32> {
    features.select(ndarray::Axis(0), indices)
}

/// Select targets by indices.
///
/// Returns a new Array1 with length `indices.len()`.
pub fn select_targets(targets: ndarray::ArrayView1<'_, f32>, indices: &[usize]) -> Array1<f32> {
    Array1::from_vec(indices.iter().map(|&i| targets[i]).collect())
}
