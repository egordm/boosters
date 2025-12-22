//! Linear model prediction extensions.
//!
//! Provides prediction methods for [`LinearModel`](crate::repr::gblinear::LinearModel).

use ndarray::Array2;
use rayon::prelude::*;

use crate::data::{FeaturesView, SamplesView};
use crate::repr::gblinear::LinearModel;

/// Extension trait for LinearModel prediction.
pub trait LinearModelPredict {
    /// Predict into a column-major buffer.
    ///
    /// Output layout: `output[group * num_rows + row]`
    ///
    /// This is the preferred method for training where predictions are stored
    /// in column-major layout for efficient gradient computation.
    fn predict_col_major(&self, data: FeaturesView<'_>, output: &mut [f32]);

    /// Predict for a single row.
    ///
    /// Returns a vector of length `num_groups`.
    fn predict_row(&self, features: &[f32], base_score: &[f32]) -> Vec<f32>;

    /// Predict for a batch of rows.
    ///
    /// Returns predictions as `Array2<f32>` with shape `(num_rows, num_groups)`.
    fn predict(&self, data: SamplesView<'_>, base_score: &[f32]) -> Array2<f32>;

    /// Parallel prediction for a batch of rows.
    ///
    /// Uses Rayon to parallelize over rows.
    /// Returns predictions as `Array2<f32>` with shape `(num_rows, num_groups)`.
    fn par_predict(&self, data: SamplesView<'_>, base_score: &[f32]) -> Array2<f32>;
}

impl LinearModelPredict for LinearModel {
    fn predict_col_major(&self, data: FeaturesView<'_>, output: &mut [f32]) {
        let num_rows = data.n_samples();
        let num_groups = self.n_groups();
        let num_features = self.n_features();
        debug_assert_eq!(output.len(), num_rows * num_groups);

        // Initialize with bias (column-major: group-first)
        for group in 0..num_groups {
            output[group * num_rows..(group + 1) * num_rows].fill(self.bias(group));
        }

        // Add weighted features - iterate over features (rows in FeaturesView)
        for feat_idx in 0..num_features {
            let feature_values = data.feature(feat_idx);
            for (row_idx, &value) in feature_values.iter().enumerate() {
                for group in 0..num_groups {
                    output[group * num_rows + row_idx] += value * self.weight(feat_idx, group);
                }
            }
        }
    }

    fn predict_row(&self, features: &[f32], base_score: &[f32]) -> Vec<f32> {
        let num_features = self.n_features();
        let num_groups = self.n_groups();

        debug_assert!(
            features.len() >= num_features,
            "not enough features: got {}, need {}",
            features.len(),
            num_features
        );

        let mut outputs = Vec::with_capacity(num_groups);

        for group in 0..num_groups {
            let base = base_score.get(group).copied().unwrap_or(0.0);
            let mut sum = base + self.bias(group);

            for (feat_idx, &value) in features.iter().take(num_features).enumerate() {
                sum += value * self.weight(feat_idx, group);
            }

            outputs.push(sum);
        }

        outputs
    }

    fn predict(&self, data: SamplesView<'_>, base_score: &[f32]) -> Array2<f32> {
        let num_rows = data.n_samples();
        let num_features = self.n_features();
        let num_groups = self.n_groups();

        // Build predictions in row-major layout: (num_rows, num_groups)
        let mut output = Array2::zeros((num_rows, num_groups));

        for row_idx in 0..num_rows {
            let row = data.sample(row_idx);

            for group in 0..num_groups {
                let base = base_score.get(group).copied().unwrap_or(0.0);
                let mut sum = base + self.bias(group);

                for feat_idx in 0..num_features {
                    let value = row.get(feat_idx).copied().unwrap_or(0.0);
                    sum += value * self.weight(feat_idx, group);
                }

                output[[row_idx, group]] = sum;
            }
        }

        output
    }

    fn par_predict(&self, data: SamplesView<'_>, base_score: &[f32]) -> Array2<f32> {
        let num_rows = data.n_samples();
        let num_features = self.n_features();
        let num_groups = self.n_groups();

        // Collect row-major results per row (each row returns its groups)
        let row_outputs: Vec<Vec<f32>> = (0..num_rows)
            .into_par_iter()
            .map(|row_idx| {
                let row = data.sample(row_idx);
                let mut row_output = Vec::with_capacity(num_groups);

                for group in 0..num_groups {
                    let base = base_score.get(group).copied().unwrap_or(0.0);
                    let mut sum = base + self.bias(group);

                    for feat_idx in 0..num_features {
                        let value = row.get(feat_idx).copied().unwrap_or(0.0);
                        sum += value * self.weight(feat_idx, group);
                    }

                    row_output.push(sum);
                }

                row_output
            })
            .collect();

        // Flatten into row-major layout
        let flat: Vec<f32> = row_outputs.into_iter().flatten().collect();
        Array2::from_shape_vec((num_rows, num_groups), flat)
            .expect("row_outputs shape mismatch")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::SamplesView;

    #[test]
    fn predict_row_regression() {
        // y = 0.5 * x0 + 0.3 * x1 + 0.1
        let weights = vec![0.5, 0.3, 0.1].into_boxed_slice();
        let model = LinearModel::new(weights, 2, 1);

        let features = vec![2.0, 3.0]; // 0.5*2 + 0.3*3 + 0.1 = 1.0 + 0.9 + 0.1 = 2.0
        let output = model.predict_row(&features, &[0.0]);

        assert_eq!(output.len(), 1);
        assert!((output[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn predict_row_with_base_score() {
        let weights = vec![0.5, 0.3, 0.1].into_boxed_slice();
        let model = LinearModel::new(weights, 2, 1);

        let features = vec![2.0, 3.0];
        let output = model.predict_row(&features, &[0.5]); // base_score = 0.5

        // 0.5 + 0.5*2 + 0.3*3 + 0.1 = 2.5
        assert!((output[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn predict_batch() {
        let weights = vec![0.5, 0.3, 0.1].into_boxed_slice();
        let model = LinearModel::new(weights, 2, 1);

        let data = [
            2.0f32, 3.0, // row 0: 0.5*2 + 0.3*3 + 0.1 = 2.0
            1.0, 1.0,    // row 1: 0.5*1 + 0.3*1 + 0.1 = 0.9
        ];
        let view = SamplesView::from_slice(&data, 2, 2).unwrap();

        let output = model.predict(view, &[0.0]);

        assert_eq!(output.dim(), (2, 1));
        assert!((output[[0, 0]] - 2.0).abs() < 1e-6);
        assert!((output[[1, 0]] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn predict_batch_multigroup() {
        // 2 features, 2 groups
        let weights = vec![
            0.1, 0.2, // feature 0
            0.3, 0.4, // feature 1
            0.0, 0.0, // bias
        ]
        .into_boxed_slice();
        let model = LinearModel::new(weights, 2, 2);

        let data = [1.0f32, 1.0];
        let view = SamplesView::from_slice(&data, 1, 2).unwrap();
        let output = model.predict(view, &[0.0, 0.0]);

        assert_eq!(output.dim(), (1, 2));
        // group 0: 0.1*1 + 0.3*1 = 0.4
        // group 1: 0.2*1 + 0.4*1 = 0.6
        assert!((output[[0, 0]] - 0.4).abs() < 1e-6);
        assert!((output[[0, 1]] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn par_predict_batch() {
        let weights = vec![0.5, 0.3, 0.1].into_boxed_slice();
        let model = LinearModel::new(weights, 2, 1);

        let data = [
            2.0f32, 3.0, // row 0
            1.0, 1.0,    // row 1
        ];
        let view = SamplesView::from_slice(&data, 2, 2).unwrap();

        let output = model.par_predict(view, &[0.0]);

        assert_eq!(output.dim(), (2, 1));
        assert!((output[[0, 0]] - 2.0).abs() < 1e-6);
        assert!((output[[1, 0]] - 0.9).abs() < 1e-6);
    }
}

