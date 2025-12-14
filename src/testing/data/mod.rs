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
pub fn synthetic_regression_targets_linear(
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

/// Generate *synthetic* binary targets (0/1) by applying a linear score + noise and thresholding at 0.
pub fn synthetic_binary_targets_from_linear_score(
	features_row_major: &[f32],
	rows: usize,
	cols: usize,
	seed: u64,
	noise_amplitude: f32,
) -> Vec<f32> {
	let (score, _w, _b) = synthetic_regression_targets_linear(features_row_major, rows, cols, seed, noise_amplitude);
	score.into_iter().map(|s| if s > 0.0 { 1.0 } else { 0.0 }).collect()
}

/// Generate *synthetic* multiclass targets (class index stored as f32) using a linear model per class.
pub fn synthetic_multiclass_targets_from_linear_scores(
	features_row_major: &[f32],
	rows: usize,
	cols: usize,
	num_classes: usize,
	seed: u64,
	noise_amplitude: f32,
) -> Vec<f32> {
	assert!(num_classes >= 2);
	assert_eq!(features_row_major.len(), rows * cols);
	let mut rng = StdRng::seed_from_u64(seed);

	let weights: Vec<f32> = (0..num_classes * cols)
		.map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
		.collect();
	let bias: Vec<f32> = (0..num_classes)
		.map(|_| rng.r#gen::<f32>() * 0.5 - 0.25)
		.collect();

	let mut labels = Vec::with_capacity(rows);
	for r in 0..rows {
		let base = r * cols;
		let mut best_class = 0usize;
		let mut best_score = f32::NEG_INFINITY;
		for k in 0..num_classes {
			let mut s = bias[k];
			let w_off = k * cols;
			for c in 0..cols {
				s += features_row_major[base + c] * weights[w_off + c];
			}
			if noise_amplitude > 0.0 {
				s += (rng.r#gen::<f32>() * 2.0 - 1.0) * noise_amplitude;
			}
			if s > best_score {
				best_score = s;
				best_class = k;
			}
		}
		labels.push(best_class as f32);
	}

	labels
}
