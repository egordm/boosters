//! Early stopping example demonstrating validation-based training termination.
//!
//! This example shows how to configure early stopping. Currently, the high-level
//! `GBDTModel::train` API monitors training loss for early stopping. For true
//! validation-based early stopping, use the lower-level `GBDTTrainer` API.
//!
//! Run with:
//! ```bash
//! cargo run --example early_stopping
//! ```

use boosters::data::binned::BinnedDatasetBuilder;
use boosters::data::{ColMatrix, DenseMatrix, RowMajor};
use boosters::{GBDTConfig, GBDTModel, Metric, Objective, TreeParams};

fn main() {
    println!("=== Early Stopping Example ===\n");

    // =========================================================================
    // 1. Prepare Data
    // =========================================================================
    let n_samples = 500;
    let n_features = 5;

    let (features, labels) = generate_regression_data(n_samples, n_features, 42);

    // Create binned dataset
    let row_matrix: DenseMatrix<f32, RowMajor> =
        DenseMatrix::from_vec(features.clone(), n_samples, n_features);
    let col_matrix: ColMatrix<f32> = row_matrix.to_layout();
    let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 256)
        .build()
        .expect("Failed to build dataset");

    // =========================================================================
    // 2. Configure Early Stopping
    // =========================================================================
    // early_stopping_rounds: Stop if metric doesn't improve for N rounds
    let config = GBDTConfig::builder()
        .n_trees(200)
        .learning_rate(0.3) // Aggressive learning rate to show early stopping
        .tree(TreeParams::depth_wise(6)) // Deep trees to overfit faster
        .objective(Objective::squared())
        .metric(Metric::rmse())
        .early_stopping_rounds(5) // Stop if no improvement for 5 rounds
        .build()
        .expect("Invalid configuration");

    println!("Configuration:");
    println!("  Max trees: {}", config.n_trees);
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Early stopping rounds: {:?}", config.early_stopping_rounds);
    println!();

    // =========================================================================
    // 3. Train with Early Stopping
    // =========================================================================
    println!("Training with early stopping (monitoring training loss)...\n");

    let model =
        GBDTModel::train(&dataset, &labels, &[], config).expect("Training failed");

    // =========================================================================
    // 4. Results
    // =========================================================================
    let actual_trees = model.forest().n_trees();
    println!("\n=== Results ===");
    println!("Trees trained: {} (max was 200)", actual_trees);

    // Evaluate
    let preds = model.predict_batch(&features, n_samples);
    let rmse = compute_rmse(&preds, &labels);

    println!("Training RMSE: {:.4}", rmse);

    if actual_trees < 200 {
        println!("\n✓ Early stopping triggered after {} trees!", actual_trees);
        println!("  Training loss stopped improving, preventing unnecessary computation.");
    } else {
        println!("\nNote: Early stopping didn't trigger because training loss");
        println!("kept improving. For validation-based early stopping, use");
        println!("the GBDTTrainer API with eval_sets parameter.");
    }
}

// Helper functions
fn generate_regression_data(n_samples: usize, n_features: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut features = Vec::with_capacity(n_samples * n_features);
    let mut labels = Vec::with_capacity(n_samples);

    // Simple deterministic pseudo-random based on seed
    let mut rng = seed;

    for _ in 0..n_samples {
        let mut row = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = ((rng >> 33) as f32) / (u32::MAX as f32) * 10.0;
            row.push(x);
        }

        // y = x0 + 0.5*x1 + 0.25*x2 + noise
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = ((rng >> 33) as f32) / (u32::MAX as f32) * 0.5 - 0.25;
        let y = row[0] + 0.5 * row[1] + 0.25 * row[2] + noise;

        features.extend(row);
        labels.push(y);
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
