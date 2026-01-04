//! Basic GBDT training example using the high-level API.
//!
//! This example demonstrates the recommended way to train a gradient boosted
//! tree model using [`GBDTModel`] and [`GBDTConfig`].
//!
//! Run with:
//! ```bash
//! cargo run --example basic_training
//! ```

use boosters::data::Dataset;
use boosters::training::GrowthStrategy;
use boosters::{GBDTConfig, GBDTModel, Metric, Objective};
use ndarray::{Array1, Array2};

fn main() {
    // =========================================================================
    // 1. Prepare Data
    // =========================================================================
    // Generate synthetic regression data: y = x0 + 0.5*x1 + 0.25*x2 + noise
    let n_samples = 500;
    let n_features = 5;

    let (features, labels) = generate_regression_data(n_samples, n_features);

    // Create training dataset (feature-major) with targets.
    let targets_2d = labels.clone().insert_axis(ndarray::Axis(0));
    let dataset = Dataset::from_array(features.view(), Some(targets_2d), None);

    // =========================================================================
    // 2. Configure and Train
    // =========================================================================
    // The high-level API uses GBDTConfig for configuration
    let config = GBDTConfig::builder()
        .n_trees(50)
        .learning_rate(0.1)
        .growth_strategy(GrowthStrategy::DepthWise { max_depth: 4 })
        .objective(Objective::SquaredLoss)
        .metric(Metric::Rmse)
        .build()
        .expect("Invalid configuration");

    println!("Training GBDT model...");
    println!("  Trees: {}", config.n_trees);
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Objective: {:?}", config.objective);
    println!("  Metric: {:?}\n", config.metric);

    // Train using GBDTModel (high-level API)
    let model = GBDTModel::train(&dataset, None, config, 1).expect("Training failed");

    // =========================================================================
    // 3. Make Predictions
    // =========================================================================
    // Predict on single sample - create feature-major array [n_features, 1]
    let first_sample_data: Vec<f32> = (0..n_features).map(|f| features[(f, 0)]).collect();
    let sample_fm = Array2::from_shape_vec((n_features, 1), first_sample_data).unwrap();
    let sample_dataset = Dataset::from_array(sample_fm.view(), None, None);
    let pred = model.predict(&sample_dataset, 1);
    println!("Sample prediction: {:.4}", pred.as_slice().unwrap()[0]);

    // Predict on full dataset - features is already feature-major
    let all_preds = model.predict(&dataset, 1);

    // Compute RMSE manually
    let rmse = compute_rmse(all_preds.as_slice().unwrap(), labels.as_slice().unwrap());

    // =========================================================================
    // 4. Inspect Model
    // =========================================================================
    println!("\n=== Model Information ===");
    println!("Trees: {}", model.forest().n_trees());
    println!("Features: {}", model.meta().n_features);
    println!("Groups: {}", model.meta().n_groups);
    println!("Train RMSE: {:.4}", rmse);

    // Training config is not stored in the trained model.
    // We keep only inference-relevant metadata like the output transform.
    println!("Output transform: {:?}", model.output_transform());

    println!("\nNote: For production, split data into train/validation/test sets!");
}

// Helper functions
fn generate_regression_data(n_samples: usize, n_features: usize) -> (Array2<f32>, Array1<f32>) {
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
