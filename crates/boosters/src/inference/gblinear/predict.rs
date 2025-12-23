//! Linear model prediction extensions.
//!
//! Provides prediction methods for [`LinearModel`](crate::repr::gblinear::LinearModel).

use ndarray::Array2;
use rayon::prelude::*;

use crate::data::{FeaturesView, SamplesView};
use crate::repr::gblinear::LinearModel;
use crate::utils::Parallelism;

/// Extension trait for LinearModel prediction.
pub trait LinearModelPredict {
    /// Predict into a column-major buffer.
    ///
    /// Output layout: `output[group * n_rows + row]`
    ///
    /// This is the preferred method for training where predictions are stored
    /// in column-major layout for efficient gradient computation.
    fn predict_col_major(&self, data: FeaturesView<'_>, output: &mut [f32]);

    /// Predict for a single row.
    ///
    /// Returns a vector of length `n_groups`.
    fn predict_row(&self, features: &[f32], base_score: &[f32]) -> Vec<f32>;

    /// Predict for a batch of rows.
    ///
    /// Returns predictions as `Array2<f32>` with shape `(n_rows, n_groups)`.
    ///
    /// # Arguments
    ///
    /// * `data` - Feature matrix (sample-major layout)
    /// * `base_score` - Base score to add per group
    fn predict(&self, data: SamplesView<'_>, base_score: &[f32]) -> Array2<f32>;

    /// Predict with explicit parallelism control.
    ///
    /// Returns predictions as `Array2<f32>` with shape `(n_rows, n_groups)`.
    ///
    /// # Arguments
    ///
    /// * `data` - Feature matrix (sample-major layout)
    /// * `base_score` - Base score to add per group
    /// * `parallelism` - Whether to use parallel execution
    fn predict_with(
        &self,
        data: SamplesView<'_>,
        base_score: &[f32],
        parallelism: Parallelism,
    ) -> Array2<f32>;
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
        self.predict_with(data, base_score, Parallelism::Sequential)
    }

    fn predict_with(
        &self,
        data: SamplesView<'_>,
        base_score: &[f32],
        parallelism: Parallelism,
    ) -> Array2<f32> {
        // Use ndarray dot product: data [n_samples, n_features] · weights [n_features, n_groups]
        // → output [n_samples, n_groups]
        let mut output = data.as_array().dot(&self.weight_view());

        // Add biases: output[row, group] += bias[group] + base_score[group]
        let biases = self.biases();

        match parallelism {
            Parallelism::Sequential => {
                for mut row in output.rows_mut() {
                    for (group, val) in row.iter_mut().enumerate() {
                        let base = base_score.get(group).copied().unwrap_or(0.0);
                        *val += biases[group] + base;
                    }
                }
            }
            Parallelism::Parallel => {
                output
                    .axis_iter_mut(ndarray::Axis(0))
                    .into_par_iter()
                    .for_each(|mut row| {
                        for (group, val) in row.iter_mut().enumerate() {
                            let base = base_score.get(group).copied().unwrap_or(0.0);
                            *val += biases[group] + base;
                        }
                    });
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::SamplesView;
    use crate::utils::Parallelism;
    use ndarray::array;

    #[test]
    fn predict_row_regression() {
        // y = 0.5 * x0 + 0.3 * x1 + 0.1
        let weights = array![[0.5], [0.3], [0.1]];
        let model = LinearModel::new(weights);

        let features = vec![2.0, 3.0]; // 0.5*2 + 0.3*3 + 0.1 = 1.0 + 0.9 + 0.1 = 2.0
        let output = model.predict_row(&features, &[0.0]);

        assert_eq!(output.len(), 1);
        assert!((output[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn predict_row_with_base_score() {
        let weights = array![[0.5], [0.3], [0.1]];
        let model = LinearModel::new(weights);

        let features = vec![2.0, 3.0];
        let output = model.predict_row(&features, &[0.5]); // base_score = 0.5

        // 0.5 + 0.5*2 + 0.3*3 + 0.1 = 2.5
        assert!((output[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn predict_batch() {
        let weights = array![[0.5], [0.3], [0.1]];
        let model = LinearModel::new(weights);

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
        let weights = array![
            [0.1, 0.2], // feature 0
            [0.3, 0.4], // feature 1
            [0.0, 0.0], // bias
        ];
        let model = LinearModel::new(weights);

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
    fn predict_with_parallelism() {
        let weights = array![[0.5], [0.3], [0.1]];
        let model = LinearModel::new(weights);

        let data = [
            2.0f32, 3.0, // row 0
            1.0, 1.0,    // row 1
        ];
        let view = SamplesView::from_slice(&data, 2, 2).unwrap();

        // Test parallel prediction
        let output = model.predict_with(view, &[0.0], Parallelism::Parallel);

        assert_eq!(output.dim(), (2, 1));
        assert!((output[[0, 0]] - 2.0).abs() < 1e-6);
        assert!((output[[1, 0]] - 0.9).abs() < 1e-6);

        // Sequential should give same results
        let seq_output = model.predict_with(view, &[0.0], Parallelism::Sequential);
        assert_eq!(output, seq_output);
    }
}

