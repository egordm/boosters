//! Custom objective function example.
//!
//! This example demonstrates how to implement a custom objective function
//! (loss function) for gradient boosting.
//!
//! Run with:
//! ```bash
//! cargo run --example custom_objective
//! ```

use boosters::data::binned::BinnedDatasetBuilder;
use boosters::data::{ColMatrix, DenseMatrix, RowMajor};
use boosters::training::{GBDTParams, GBDTTrainer, GradsTuple, GrowthStrategy, Rmse, TargetSchema};
use boosters::{ObjectiveFn, Parallelism, TaskKind};
use boosters::inference::common::PredictionKind;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut2};

fn empty_weights() -> ArrayView1<'static, f32> {
    ArrayView1::from(&[][..])
}

/// A custom objective: Huber loss with delta=1.0
///
/// Huber loss combines the best of MSE (smooth near zero) and MAE (robust to outliers).
/// - For |error| <= delta: loss = 0.5 * error^2 (like MSE)
/// - For |error| > delta: loss = delta * (|error| - 0.5 * delta) (like MAE)
///
/// Gradients:
/// - For |error| <= delta: grad = error, hess = 1.0
/// - For |error| > delta: grad = delta * sign(error), hess = 0.0 (or small epsilon)
#[derive(Clone, Debug)]
pub struct HuberLoss {
    delta: f32,
}

impl HuberLoss {
    pub fn new(delta: f32) -> Self {
        Self { delta }
    }
}

impl ObjectiveFn for HuberLoss {
    fn n_outputs(&self) -> usize {
        1
    }

    fn compute_gradients(
        &self,
        predictions: ArrayView2<f32>,
        targets: ArrayView1<f32>,
        weights: ArrayView1<f32>,
        mut grad_hess: ArrayViewMut2<GradsTuple>,
    ) {
        let (n_outputs, _n_rows) = predictions.dim();
        let weights_slice = weights.as_slice().unwrap_or(&[]);
        let targets_slice = targets.as_slice().unwrap_or(&[]);

        for out_idx in 0..n_outputs {
            let preds_row = predictions.row(out_idx);
            let preds_slice = preds_row.as_slice().unwrap();
            let mut gh_row = grad_hess.row_mut(out_idx);
            let gh_slice = gh_row.as_slice_mut().unwrap();

            for (i, (gh, &pred)) in gh_slice.iter_mut().zip(preds_slice.iter()).enumerate() {
                let target = targets_slice[i];
                let w = if weights_slice.is_empty() { 1.0 } else { weights_slice[i] };
                let error = pred - target;

                let (grad, hess) = if error.abs() <= self.delta {
                    // Quadratic region
                    (w * error, w)
                } else {
                    // Linear region
                    (w * self.delta * error.signum(), w * 1e-6) // Small hess for stability
                };

                gh.grad = grad;
                gh.hess = hess;
            }
        }
    }

    fn compute_base_score(
        &self,
        targets: ArrayView1<f32>,
        _weights: ArrayView1<f32>,
        mut outputs: ArrayViewMut2<f32>,
    ) {
        // Use median for Huber (more robust than mean)
        // For simplicity, using mean here
        let n_rows = targets.len();
        if n_rows == 0 {
            outputs.fill(0.0);
            return;
        }

        let targets_slice = targets.as_slice().unwrap_or(&[]);
        let sum: f32 = targets_slice.iter().sum();
        outputs.fill(sum / n_rows as f32);
    }

    fn name(&self) -> &'static str {
        "huber"
    }

    fn task_kind(&self) -> TaskKind {
        TaskKind::Regression
    }

    fn target_schema(&self) -> TargetSchema {
        TargetSchema::Continuous
    }

    fn transform_predictions(&self, _predictions: ArrayViewMut2<f32>) -> PredictionKind {
        PredictionKind::Value
    }
}

fn main() {
    // =========================================================================
    // 1. Generate Data with Outliers
    // =========================================================================
    let n_samples = 500;
    let n_features = 5;

    let (features, labels) = generate_data_with_outliers(n_samples, n_features);

    // Create binned dataset
    let row_matrix: DenseMatrix<f32, RowMajor> =
        DenseMatrix::from_vec(features.clone(), n_samples, n_features);
    let col_matrix: ColMatrix<f32> = row_matrix.to_layout();
    let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 256)
        .build()
        .expect("Failed to build binned dataset");

    // =========================================================================
    // 2. Train with Custom Objective
    // =========================================================================
    let params = GBDTParams {
        n_trees: 50,
        learning_rate: 0.1,
        growth_strategy: GrowthStrategy::DepthWise { max_depth: 4 },
        ..Default::default()
    };

    println!("Training with custom Huber loss (delta=1.0)...\n");

    let huber = HuberLoss::new(1.0);
    let trainer = GBDTTrainer::new(huber, Rmse, params);
    let forest = trainer
        .train(&dataset, ArrayView1::from(&labels[..]), empty_weights(), &[], Parallelism::Sequential)
        .unwrap();

    // =========================================================================
    // 3. Evaluate
    // =========================================================================
    let predictions: Vec<f32> = features
        .chunks(n_features)
        .map(|row| forest.predict_row(row)[0])
        .collect();

    let rmse = compute_rmse(&predictions, &labels);
    let mae = compute_mae(&predictions, &labels);

    println!("=== Results ===");
    println!("Trees: {}", forest.n_trees());
    println!("RMSE: {:.4}", rmse);
    println!("MAE:  {:.4}", mae);
    println!("\nHuber loss is robust to outliers, so MAE may be better than pure MSE training.");
}

// Helper: generate data with outliers
fn generate_data_with_outliers(n_samples: usize, n_features: usize) -> (Vec<f32>, Vec<f32>) {
    let mut features = Vec::with_capacity(n_samples * n_features);
    let mut labels = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let x0 = (i as f32) / (n_samples as f32) * 10.0;
        let x1 = ((i * 7) % 100) as f32 / 10.0;
        let x2 = ((i * 13) % 100) as f32 / 10.0;
        let x3 = ((i * 17) % 100) as f32 / 10.0;
        let x4 = ((i * 23) % 100) as f32 / 10.0;

        features.push(x0);
        features.push(x1);
        features.push(x2);
        features.push(x3);
        features.push(x4);

        // Add outliers every 50 samples
        let label = if i % 50 == 0 {
            x0 + 0.5 * x1 + 100.0 // Outlier: large offset
        } else {
            x0 + 0.5 * x1 + 0.25 * x2
        };
        labels.push(label);
    }

    (features, labels)
}

fn compute_rmse(predictions: &[f32], labels: &[f32]) -> f64 {
    let mse: f64 = predictions
        .iter()
        .zip(labels.iter())
        .map(|(p, l)| {
            let diff = *p as f64 - *l as f64;
            diff * diff
        })
        .sum::<f64>()
        / labels.len() as f64;
    mse.sqrt()
}

fn compute_mae(predictions: &[f32], labels: &[f32]) -> f64 {
    predictions
        .iter()
        .zip(labels.iter())
        .map(|(p, l)| (*p as f64 - *l as f64).abs())
        .sum::<f64>()
        / labels.len() as f64
}
