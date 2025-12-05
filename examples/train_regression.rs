//! GBTree regression training example.
//!
//! This example demonstrates training a gradient boosted tree model for regression.
//!
//! Run with:
//! ```bash
//! cargo run --example train_regression
//! ```

use booste_rs::data::{ColMatrix, RowMatrix};
use booste_rs::predict::{Predictor, StandardTraversal};
use booste_rs::training::quantize::ExactQuantileCuts;
use booste_rs::training::{
    DepthWisePolicy, GBTreeTrainer, Quantizer, SquaredLoss, TrainerParams, TreeParams, Verbosity,
};

fn main() {
    // Generate synthetic regression data
    // y = x0 + 0.5*x1 + 0.25*x2 + noise
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

        // Target with some noise
        let noise = ((i * 31) % 100) as f32 / 500.0 - 0.1;
        let y = x0 + 0.5 * x1 + 0.25 * x2 + noise;
        labels.push(y);
    }

    // Create matrix (row-major input)
    let row_matrix = RowMatrix::from_vec(features, n_samples, n_features);
    // Convert to column-major for training (required by quantization)
    let col_matrix: ColMatrix<f32> = row_matrix.to_layout();

    // Quantize features (required for histogram-based training)
    let cut_finder = ExactQuantileCuts::default();
    let quantizer = Quantizer::from_data(&col_matrix, &cut_finder, 256);
    let cuts = quantizer.cuts().clone();
    let quantized = quantizer.quantize::<_, u8>(&col_matrix);

    // Configure training parameters
    let tree_params = TreeParams {
        max_depth: 4,
        learning_rate: 0.1,
        ..Default::default()
    };

    let params = TrainerParams {
        num_rounds: 50,
        tree_params,
        verbosity: Verbosity::Info, // Show training progress
        ..Default::default()
    };

    // Create trainer with squared loss (regression)
    let mut trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params);

    // Use depth-wise growth strategy (like XGBoost default)
    let policy = DepthWisePolicy { max_depth: 4 };

    // Train!
    println!("Training GBTree regression model...\n");
    let forest = trainer.train(policy, &quantized, &labels, &cuts, &[]);

    // Create predictor from trained forest
    let predictor = Predictor::<StandardTraversal>::new(&forest);

    // Evaluate on training set
    let train_preds = predictor.predict(&row_matrix).into_vec();
    let train_rmse = compute_rmse(&train_preds, &labels);

    println!("\n=== Results ===");
    println!("Trees: {}", forest.num_trees());
    println!("Train RMSE: {:.4}", train_rmse);
    println!("\nNote: For production, split data into train/validation/test sets!");
}

/// Compute Root Mean Squared Error.
fn compute_rmse(predictions: &[f32], labels: &[f32]) -> f32 {
    let mse: f32 = predictions
        .iter()
        .zip(labels.iter())
        .map(|(&p, &l)| (p - l).powi(2))
        .sum::<f32>()
        / predictions.len() as f32;
    mse.sqrt()
}
