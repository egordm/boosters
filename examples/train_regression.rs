//! GBTree regression training example.
//!
//! This example demonstrates training a gradient boosted tree model for regression.
//!
//! Run with:
//! ```bash
//! cargo run --example train_regression
//! ```

use booste_rs::data::binned::BinnedDatasetBuilder;
use booste_rs::data::{ColMatrix, DenseMatrix, RowMajor};
use booste_rs::training::{GBDTParams, GBDTTrainer, GrowthStrategy, Metric, Rmse, SquaredLoss};

fn main() {
    // =========================================================================
    // Generate synthetic regression data: y = x0 + 0.5*x1 + 0.25*x2 + noise
    // =========================================================================
    let n_samples = 500;
    let n_features = 5;

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

    // Create binned dataset for training
    let row_matrix: DenseMatrix<f32, RowMajor> =
        DenseMatrix::from_vec(features.clone(), n_samples, n_features);
    let col_matrix: ColMatrix<f32> = row_matrix.to_layout();
    let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 256)
        .build()
        .expect("Failed to build binned dataset");

    // =========================================================================
    // Train
    // =========================================================================
    let params = GBDTParams {
        n_trees: 50,
        learning_rate: 0.1,
        growth_strategy: GrowthStrategy::DepthWise { max_depth: 4 },
        cache_size: 64,
        ..Default::default()
    };

    println!("Training GBTree regression model...");
    println!("  Trees: {}", params.n_trees);
    println!("  Learning rate: {}", params.learning_rate);
    println!("  Growth: {:?}\n", params.growth_strategy);

    let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);
    let forest = trainer.train(&dataset, &labels, &[], &[]).unwrap();

    // =========================================================================
    // Evaluate
    // =========================================================================
    let predictions: Vec<f32> = features
        .chunks(n_features)
        .map(|row| forest.predict_row(row)[0])
        .collect();

    let rmse = Rmse.compute(n_samples, 1, &predictions, &labels, &[]);

    println!("=== Results ===");
    println!("Trees: {}", forest.n_trees());
    println!("Train RMSE: {:.4}", rmse);
    println!("\nNote: For production, split data into train/validation/test sets!");
}
