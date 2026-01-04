//! Custom objective function example.
//!
//! This example demonstrates how to implement a custom objective function
//! (loss function) for gradient boosting.
//!
//! Run with:
//! ```bash
//! cargo run --example custom_objective
//! ```

use boosters::data::BinningConfig;
use boosters::data::binned::BinnedDataset;
use boosters::data::{Dataset, TargetsView, WeightsView};
use boosters::inference::gbdt::SimplePredictor;
use boosters::repr::gbdt::Forest;
use boosters::training::{
    CustomObjective, GBDTParams, GBDTTrainer, GradsTuple, GrowthStrategy, Metric, Objective,
};
use boosters::{OutputTransform, Parallelism};
use ndarray::Array2;

/// Predict a single row using the predictor.
fn predict_row(forest: &Forest, features: &[f32]) -> Vec<f32> {
    let predictor = SimplePredictor::new(forest);
    let mut output = vec![0.0; predictor.n_groups()];
    let sample = ndarray::ArrayView1::from(features);
    predictor.predict_row_into(sample, None, &mut output);
    output
}

/// A custom objective: Huber loss.
///
/// Huber loss combines the best of MSE (smooth near zero) and MAE (robust to outliers).
/// - For |error| <= delta: grad = error, hess = 1.0
/// - For |error| > delta: grad = delta * sign(error), hess = small epsilon
fn huber_objective(delta: f32) -> Objective {
    let custom = CustomObjective::new(
        format!("huber(delta={delta})"),
        move |preds, targets, weights, mut grad_hess| {
            let n_rows = targets.n_samples();
            let targets_row = targets.output(0);

            for (i, w) in weights.iter(n_rows).enumerate() {
                let pred = preds[[0, i]];
                let target = targets_row[i];
                let error = pred - target;

                let (grad, hess) = if error.abs() <= delta {
                    (w * error, w)
                } else {
                    (w * delta * error.signum(), w * 1e-6)
                };

                grad_hess[[0, i]] = GradsTuple { grad, hess };
            }
        },
        |targets, weights| {
            let n_rows = targets.n_samples();
            if n_rows == 0 {
                return vec![0.0];
            }

            let targets_row = targets.output(0);
            let mut sum_w = 0.0f32;
            let mut sum_y = 0.0f32;

            for (y, w) in targets_row.iter().zip(weights.iter(n_rows)) {
                sum_w += w;
                sum_y += w * *y;
            }

            if sum_w == 0.0 {
                vec![0.0]
            } else {
                vec![sum_y / sum_w]
            }
        },
        OutputTransform::Identity,
        1,
    );

    Objective::Custom(custom)
}

fn main() {
    // =========================================================================
    // 1. Generate Data with Outliers
    // =========================================================================
    let n_samples = 500;
    let n_features = 5;

    let (features, labels) = generate_data_with_outliers(n_samples, n_features);

    // Create binned dataset
    let features_dataset = Dataset::from_array(features.view(), None, None);
    let binning_config = BinningConfig::builder().max_bins(256).build();
    let dataset =
        BinnedDataset::from_dataset(&features_dataset, &binning_config).expect("binning failed");

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

    // Wrap labels in TargetsView (shape [n_outputs=1, n_samples])
    let targets_2d = ndarray::Array2::from_shape_vec((1, labels.len()), labels.clone()).unwrap();
    let targets = TargetsView::new(targets_2d.view());

    let objective = huber_objective(1.0);
    let trainer = GBDTTrainer::new(objective, Metric::Rmse, params);
    let forest = trainer
        .train(
            &features_dataset,
            &dataset,
            targets,
            WeightsView::None,
            None,
            Parallelism::Sequential,
        )
        .unwrap();

    // =========================================================================
    // 3. Evaluate
    // =========================================================================
    // Predict row by row (features is feature-major, need to access columns as samples)
    let predictions: Vec<f32> = (0..n_samples)
        .map(|i| {
            let row: Vec<f32> = (0..n_features).map(|f| features[(f, i)]).collect();
            predict_row(&forest, &row)[0]
        })
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
fn generate_data_with_outliers(n_samples: usize, n_features: usize) -> (Array2<f32>, Vec<f32>) {
    // Feature-major data [n_features, n_samples]
    let mut features = Array2::<f32>::zeros((n_features, n_samples));
    let mut labels = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let x0 = (i as f32) / (n_samples as f32) * 10.0;
        let x1 = ((i * 7) % 100) as f32 / 10.0;
        let x2 = ((i * 13) % 100) as f32 / 10.0;
        let x3 = ((i * 17) % 100) as f32 / 10.0;
        let x4 = ((i * 23) % 100) as f32 / 10.0;

        features[(0, i)] = x0;
        features[(1, i)] = x1;
        features[(2, i)] = x2;
        features[(3, i)] = x3;
        features[(4, i)] = x4;

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
