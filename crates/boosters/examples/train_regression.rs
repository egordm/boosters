//! GBTree regression training example.
//!
//! This example demonstrates training a gradient boosted tree model for regression.
//!
//! Run with:
//! ```bash
//! cargo run --example train_regression
//! ```

use boosters::data::Dataset;
use boosters::training::GrowthStrategy;
use boosters::{GBDTConfig, GBDTModel, Metric, Objective};
use ndarray::{Array1, Array2};

fn main() {
    // =========================================================================
    // Generate synthetic regression data: y = x0 + 0.5*x1 + 0.25*x2 + noise
    // =========================================================================
    let n_samples = 500;
    let n_features = 5;

    // Generate feature-major data [n_features, n_samples]
    let mut features = Array2::<f32>::zeros((n_features, n_samples));
    let mut labels = Array1::<f32>::zeros(n_samples);

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

        let noise = ((i * 31) % 100) as f32 / 500.0 - 0.1;
        labels[i] = x0 + 0.5 * x1 + 0.25 * x2 + noise;
    }

    // Create training dataset (feature-major) with targets.
    let targets_2d = labels.clone().insert_axis(ndarray::Axis(0));
    let dataset = Dataset::from_array(features.view(), Some(targets_2d), None);

    // =========================================================================
    // Train
    // =========================================================================
    let config = GBDTConfig::builder()
        .objective(Objective::SquaredLoss)
        .metric(Metric::Rmse)
        .n_trees(50)
        .learning_rate(0.1)
        .growth_strategy(GrowthStrategy::DepthWise { max_depth: 4 })
        .cache_size(64)
        .build()
        .expect("Invalid configuration");

    println!("Training GBTree regression model...");
    println!("  Trees: {}", config.n_trees);
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Growth: {:?}\n", config.growth_strategy);

    let model = GBDTModel::train(&dataset, None, config, 1).expect("Training failed");

    // =========================================================================
    // Evaluate
    // =========================================================================
    let predictions = model.predict(&dataset, 1);

    let rmse = compute_rmse(predictions.as_slice().unwrap(), labels.as_slice().unwrap());

    println!("=== Results ===");
    println!("Trees: {}", model.forest().n_trees());
    println!("Train RMSE: {:.4}", rmse);
    println!("\nNote: For production, split data into train/validation/test sets!");
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
