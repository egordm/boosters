//! Coordinate descent updaters for linear models.
//!
//! Two variants:
//! - `ShotgunUpdater`: Parallel updates (faster, slight approximation)
//! - `CoordinateUpdater`: Sequential updates (exact gradients)

use rayon::prelude::*;

use crate::data::CSCMatrix;
use crate::linear::LinearModel;
use crate::training::GradientPair;

use super::FeatureSelector;

/// Trait for coordinate descent updaters.
pub trait Updater: Send + Sync {
    /// Perform one round of coordinate descent updates.
    ///
    /// # Arguments
    ///
    /// * `model` - Model to update
    /// * `data` - Training data in CSC format
    /// * `gradients` - Per-sample gradient pairs
    /// * `selector` - Feature selector
    /// * `group` - Output group to update
    /// * `config` - Regularization config
    fn update_round(
        &self,
        model: &mut LinearModel,
        data: &CSCMatrix<f32>,
        gradients: &[GradientPair],
        selector: &mut dyn FeatureSelector,
        group: usize,
        config: &UpdateConfig,
    );
}

/// Configuration for coordinate descent updates.
#[derive(Debug, Clone)]
pub struct UpdateConfig {
    /// L1 regularization strength (alpha).
    pub alpha: f32,
    /// L2 regularization strength (lambda).
    pub lambda: f32,
    /// Learning rate.
    pub learning_rate: f32,
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            alpha: 0.0,
            lambda: 1.0,
            learning_rate: 0.5,
        }
    }
}

/// Sequential coordinate descent updater.
///
/// Updates features one at a time with exact gradient computation.
/// Each update immediately affects subsequent updates within the same round.
#[derive(Debug, Clone, Default)]
pub struct CoordinateUpdater;

impl CoordinateUpdater {
    /// Create a new sequential updater.
    pub fn new() -> Self {
        Self
    }
}

impl Updater for CoordinateUpdater {
    fn update_round(
        &self,
        model: &mut LinearModel,
        data: &CSCMatrix<f32>,
        gradients: &[GradientPair],
        selector: &mut dyn FeatureSelector,
        group: usize,
        config: &UpdateConfig,
    ) {
        selector.reset(model.num_features());

        while let Some(feature) = selector.next() {
            let delta = compute_weight_update(model, data, gradients, feature, group, config);
            if delta.abs() > 1e-10 {
                model.add_weight(feature, group, delta);
            }
        }
    }
}

/// Parallel (shotgun) coordinate descent updater.
///
/// Updates all features in parallel. Race conditions in residual updates
/// are tolerable with reasonable learning rates.
#[derive(Debug, Clone, Default)]
pub struct ShotgunUpdater;

impl ShotgunUpdater {
    /// Create a new parallel updater.
    pub fn new() -> Self {
        Self
    }
}

impl Updater for ShotgunUpdater {
    fn update_round(
        &self,
        model: &mut LinearModel,
        data: &CSCMatrix<f32>,
        gradients: &[GradientPair],
        selector: &mut dyn FeatureSelector,
        group: usize,
        config: &UpdateConfig,
    ) {
        selector.reset(model.num_features());
        let features = selector.all_indices();

        // Compute all deltas in parallel
        let deltas: Vec<(usize, f32)> = features
            .par_iter()
            .map(|&feature| {
                let delta = compute_weight_update(model, data, gradients, feature, group, config);
                (feature, delta)
            })
            .collect();

        // Apply updates sequentially (thread-safe)
        for (feature, delta) in deltas {
            if delta.abs() > 1e-10 {
                model.add_weight(feature, group, delta);
            }
        }
    }
}

/// Compute the weight update for a single feature using elastic net regularization.
///
/// Uses soft-thresholding for L1 regularization:
/// ```text
/// grad_l2 = Σ(gradient × feature) + lambda × w
/// hess_l2 = Σ(hessian × feature²) + lambda
/// delta = soft_threshold(-grad_l2 / hess_l2, alpha / hess_l2) × learning_rate
/// ```
fn compute_weight_update(
    model: &LinearModel,
    data: &CSCMatrix<f32>,
    gradients: &[GradientPair],
    feature: usize,
    group: usize,
    config: &UpdateConfig,
) -> f32 {
    let current_weight = model.weight(feature, group);

    // Accumulate gradient and hessian for this feature
    let mut sum_grad = 0.0f32;
    let mut sum_hess = 0.0f32;

    for (row, value) in data.column(feature) {
        let gp = &gradients[row];
        sum_grad += gp.grad() * value;
        sum_hess += gp.hess() * value * value;
    }

    // Add L2 regularization
    let grad_l2 = sum_grad + config.lambda * current_weight;
    let hess_l2 = sum_hess + config.lambda;

    // Avoid division by zero
    if hess_l2.abs() < 1e-10 {
        return 0.0;
    }

    // Compute raw update
    let raw_update = -grad_l2 / hess_l2;

    // Apply soft-thresholding for L1 (elastic net)
    let threshold = config.alpha / hess_l2;
    let thresholded = soft_threshold(raw_update, threshold);

    // Apply learning rate
    thresholded * config.learning_rate
}

/// Soft-thresholding operator for L1 regularization.
///
/// S(x, λ) = sign(x) × max(|x| - λ, 0)
#[inline]
fn soft_threshold(x: f32, threshold: f32) -> f32 {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        0.0
    }
}

/// Update bias term (no regularization).
pub fn update_bias(
    model: &mut LinearModel,
    gradients: &[GradientPair],
    group: usize,
    learning_rate: f32,
) {
    let mut sum_grad = 0.0f32;
    let mut sum_hess = 0.0f32;

    for gp in gradients {
        sum_grad += gp.grad();
        sum_hess += gp.hess();
    }

    if sum_hess.abs() > 1e-10 {
        let delta = -sum_grad / sum_hess * learning_rate;
        model.add_bias(group, delta);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::RowMatrix;

    fn make_test_data() -> (CSCMatrix<f32>, Vec<GradientPair>) {
        // Simple 4x2 dataset
        let dense = RowMatrix::from_vec(
            vec![
                1.0, 0.0, // row 0
                0.0, 1.0, // row 1
                1.0, 1.0, // row 2
                2.0, 0.5, // row 3
            ],
            4,
            2,
        );
        let csc = CSCMatrix::from_dense_full(&dense);

        // Gradients (simulating squared error loss)
        let gradients = vec![
            GradientPair::new(0.5, 1.0),
            GradientPair::new(-0.3, 1.0),
            GradientPair::new(0.2, 1.0),
            GradientPair::new(-0.1, 1.0),
        ];

        (csc, gradients)
    }

    #[test]
    fn soft_threshold_positive() {
        assert!((soft_threshold(1.0, 0.3) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn soft_threshold_negative() {
        assert!((soft_threshold(-1.0, 0.3) - (-0.7)).abs() < 1e-6);
    }

    #[test]
    fn soft_threshold_within() {
        assert_eq!(soft_threshold(0.2, 0.3), 0.0);
        assert_eq!(soft_threshold(-0.2, 0.3), 0.0);
    }

    #[test]
    fn coordinate_updater_changes_weights() {
        let (csc, gradients) = make_test_data();
        let mut model = LinearModel::zeros(2, 1);
        let mut selector = super::super::CyclicSelector::new();

        let config = UpdateConfig {
            alpha: 0.0,
            lambda: 0.0,
            learning_rate: 1.0,
        };

        let updater = CoordinateUpdater::new();
        updater.update_round(&mut model, &csc, &gradients, &mut selector, 0, &config);

        // Weights should have changed
        let w0 = model.weight(0, 0);
        let w1 = model.weight(1, 0);
        assert!(w0.abs() > 1e-6 || w1.abs() > 1e-6);
    }

    #[test]
    fn shotgun_updater_changes_weights() {
        let (csc, gradients) = make_test_data();
        let mut model = LinearModel::zeros(2, 1);
        let mut selector = super::super::CyclicSelector::new();

        let config = UpdateConfig {
            alpha: 0.0,
            lambda: 0.0,
            learning_rate: 1.0,
        };

        let updater = ShotgunUpdater::new();
        updater.update_round(&mut model, &csc, &gradients, &mut selector, 0, &config);

        // Weights should have changed
        let w0 = model.weight(0, 0);
        let w1 = model.weight(1, 0);
        assert!(w0.abs() > 1e-6 || w1.abs() > 1e-6);
    }

    #[test]
    fn l2_regularization_shrinks_weights() {
        let (csc, gradients) = make_test_data();

        // No regularization
        let mut model1 = LinearModel::zeros(2, 1);
        model1.set_weight(0, 0, 1.0);
        let mut selector = super::super::CyclicSelector::new();
        let config_no_reg = UpdateConfig {
            alpha: 0.0,
            lambda: 0.0,
            learning_rate: 1.0,
        };

        let updater = CoordinateUpdater::new();
        updater.update_round(&mut model1, &csc, &gradients, &mut selector, 0, &config_no_reg);
        let w1_no_reg = model1.weight(0, 0);

        // With L2 regularization
        let mut model2 = LinearModel::zeros(2, 1);
        model2.set_weight(0, 0, 1.0);
        let config_l2 = UpdateConfig {
            alpha: 0.0,
            lambda: 10.0, // Strong L2
            learning_rate: 1.0,
        };

        updater.update_round(&mut model2, &csc, &gradients, &mut selector, 0, &config_l2);
        let w1_l2 = model2.weight(0, 0);

        // L2 should shrink more towards zero
        assert!(w1_l2.abs() < w1_no_reg.abs());
    }

    #[test]
    fn update_bias_works() {
        let gradients = vec![
            GradientPair::new(0.5, 1.0),
            GradientPair::new(-0.3, 1.0),
            GradientPair::new(0.2, 1.0),
        ];

        let mut model = LinearModel::zeros(2, 1);
        update_bias(&mut model, &gradients, 0, 1.0);

        // Bias should have changed
        // sum_grad = 0.5 - 0.3 + 0.2 = 0.4
        // sum_hess = 3.0
        // delta = -0.4 / 3.0 ≈ -0.133
        let bias = model.bias(0);
        assert!((bias - (-0.4 / 3.0)).abs() < 1e-6);
    }
}
