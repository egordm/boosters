use rand::prelude::*;

use crate::data::{DenseMatrix, RowMajor, RowMatrix};

/// Generate random dense features in row-major order.
///
/// Values are uniform in `[min, max]`.
pub fn random_dense_f32(rows: usize, cols: usize, seed: u64, min: f32, max: f32) -> Vec<f32> {
	assert!(max >= min);
	let mut rng = StdRng::seed_from_u64(seed);
	let width = max - min;
	(0..rows * cols)
		.map(|_| min + rng.r#gen::<f32>() * width)
		.collect()
}

/// Create a [`RowMatrix`] from random dense features.
pub fn random_row_matrix_f32(rows: usize, cols: usize, seed: u64, min: f32, max: f32) -> RowMatrix<f32> {
	let data = random_dense_f32(rows, cols, seed, min, max);
	DenseMatrix::<f32, RowMajor>::from_vec(data, rows, cols)
}

/// Generate regression targets as a simple linear model of features plus uniform noise.
///
/// Returns `(targets, weights, bias)`.
pub fn regression_targets_linear(
	features_row_major: &[f32],
	rows: usize,
	cols: usize,
	seed: u64,
	noise_amplitude: f32,
) -> (Vec<f32>, Vec<f32>, f32) {
	assert_eq!(features_row_major.len(), rows * cols);
	let mut rng = StdRng::seed_from_u64(seed);

	let weights: Vec<f32> = (0..cols).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect();
	let bias: f32 = rng.r#gen::<f32>() * 0.5 - 0.25;

	let mut targets = Vec::with_capacity(rows);
	for r in 0..rows {
		let mut y = bias;
		let base = r * cols;
		for c in 0..cols {
			y += features_row_major[base + c] * weights[c];
		}
		if noise_amplitude > 0.0 {
			y += (rng.r#gen::<f32>() * 2.0 - 1.0) * noise_amplitude;
		}
		targets.push(y);
	}

	(targets, weights, bias)
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
