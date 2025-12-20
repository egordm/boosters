//! Basic GBDT training example using the high-level API.
//!
//! This example demonstrates the recommended way to train a gradient boosted
//! tree model using [`GBDTModel`] and [`GBDTConfig`].
//!
//! Run with:
//! ```bash
//! cargo run --example basic_training
//! ```

use boosters::data::binned::BinnedDatasetBuilder;
use boosters::data::{ColMatrix, DenseMatrix, RowMajor};
use boosters::{GBDTConfig, GBDTModel, Metric, Objective, TreeParams};

fn main() {
    // =========================================================================
    // 1. Prepare Data
    // =========================================================================
    // Generate synthetic regression data: y = x0 + 0.5*x1 + 0.25*x2 + noise
    let n_samples = 500;
    let n_features = 5;

    let (features, labels) = generate_regression_data(n_samples, n_features);

    // Create binned dataset for training
    let row_matrix: DenseMatrix<f32, RowMajor> =
        DenseMatrix::from_vec(features.clone(), n_samples, n_features);
    let col_matrix: ColMatrix<f32> = row_matrix.to_layout();
    let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 256)
        .build()
        .expect("Failed to build binned dataset");

    // =========================================================================
    // 2. Configure and Train
    // =========================================================================
    // The high-level API uses GBDTConfig for configuration
    let config = GBDTConfig::builder()
        .n_trees(50)
        .learning_rate(0.1)
        .tree(TreeParams::depth_wise(4))
        .objective(Objective::squared())
        .metric(Metric::rmse())
        .build()
        .expect("Invalid configuration");

    println!("Training GBDT model...");
    println!("  Trees: {}", config.n_trees);
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Objective: {:?}", config.objective);
    println!("  Metric: {:?}\n", config.metric);

    // Train using GBDTModel (high-level API)
    let model = GBDTModel::train(&dataset, &labels, &[], config).expect("Training failed");

    // =========================================================================
    // 3. Make Predictions
    // =========================================================================
    // Predict on single sample (wrap slice in a 1-row matrix)
    let sample: DenseMatrix<f32, RowMajor> =
        DenseMatrix::from_vec(features[0..n_features].to_vec(), 1, n_features);
    let pred = model.predict(&sample);
    println!("Sample prediction: {:.4}", pred.as_slice()[0]);

    // Predict on full dataset (returns ColMatrix)
    let all_preds = model.predict(&row_matrix);

    // Compute RMSE manually
    let rmse = compute_rmse(all_preds.as_slice(), &labels);

    // =========================================================================
    // 4. Inspect Model
    // =========================================================================
    println!("\n=== Model Information ===");
    println!("Trees: {}", model.forest().n_trees());
    println!("Features: {}", model.meta().n_features);
    println!("Task: {:?}", model.meta().task);
    println!("Train RMSE: {:.4}", rmse);

    // Access training config if available
    if let Some(config) = model.config() {
        println!("Learning rate used: {}", config.learning_rate);
    }

    println!("\nNote: For production, split data into train/validation/test sets!");
}

// Helper functions
fn generate_regression_data(n_samples: usize, n_features: usize) -> (Vec<f32>, Vec<f32>) {
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

        let noise = ((i * 31) % 100) as f32 / 500.0 - 0.1;
        labels.push(x0 + 0.5 * x1 + 0.25 * x2 + noise);
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
