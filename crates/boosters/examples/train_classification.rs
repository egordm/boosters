//! GBTree binary classification training example.
//!
//! This example demonstrates training a gradient boosted tree model for
//! binary classification using logistic loss.
//!
//! ## Features Shown
//!
//! - Binary classification with `LogisticLoss`
//! - Depth-wise vs Leaf-wise tree growth strategies
//! - Sample weighting for imbalanced data
//!
//! For a more detailed example of handling class imbalance, see
//! `train_imbalanced.rs`.
//!
//! Run with:
//! ```bash
//! cargo run --example train_classification
//! ```

use boosters::data::binned::BinnedDatasetBuilder;
use boosters::data::{ColMatrix, DenseMatrix, RowMajor};
use boosters::{GBDTConfig, GBDTModel, Metric, Objective, TreeParams};
use ndarray::ArrayView1;

fn main() {
    // =========================================================================
    // Generate synthetic binary classification data
    // =========================================================================
    // Two clusters: class 0 centered at (2, 2), class 1 centered at (8, 8)
    let n_samples = 400;
    let n_features = 4;

    let mut features = Vec::with_capacity(n_samples * n_features);
    let mut labels = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let class = (i % 2) as f32;
        let offset = if class == 0.0 { 2.0 } else { 8.0 };

        let noise1 = ((i * 17) % 100) as f32 / 50.0 - 1.0;
        let noise2 = ((i * 23) % 100) as f32 / 50.0 - 1.0;
        let noise3 = ((i * 31) % 100) as f32 / 50.0 - 1.0;
        let noise4 = ((i * 37) % 100) as f32 / 50.0 - 1.0;

        features.push(offset + noise1);
        features.push(offset + noise2);
        features.push(noise3 * 2.0);
        features.push(noise4 * 2.0);

        labels.push(class);
    }

    // Create binned dataset for training
    let row_matrix: DenseMatrix<f32, RowMajor> =
        DenseMatrix::from_vec(features.clone(), n_samples, n_features);
    let col_matrix: ColMatrix<f32> = row_matrix.to_layout();

    // Create binned dataset
    let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 256)
        .build()
        .expect("Failed to build binned dataset");

    // =========================================================================
    // Train with depth-wise growth (XGBoost style)
    // =========================================================================
    println!("=== Depth-wise Growth (XGBoost style) ===\n");

    let config_depth = GBDTConfig::builder()
        .objective(Objective::logistic())
        .metric(Metric::logloss())
        .n_trees(30)
        .learning_rate(0.1)
        .tree(TreeParams::depth_wise(3))
        .cache_size(32)
        .build()
        .expect("Invalid configuration");

    let model_depth = GBDTModel::train(&dataset, ArrayView1::from(&labels[..]), None, config_depth, 1)
        .expect("Training failed");

    // Predict: GBDTModel::predict() returns probabilities for logistic objective
    let predictions = model_depth.predict(&row_matrix, 1);

    let acc = compute_accuracy(predictions.as_slice(), &labels);
    println!("Depth-wise: {} trees", model_depth.forest().n_trees());
    println!("Accuracy: {:.2}%", acc * 100.0);

    // =========================================================================
    // Train with leaf-wise growth (LightGBM style)
    // =========================================================================
    println!("\n=== Leaf-wise Growth (LightGBM style) ===\n");

    let config_leaf = GBDTConfig::builder()
        .objective(Objective::logistic())
        .metric(Metric::logloss())
        .n_trees(30)
        .learning_rate(0.1)
        .tree(TreeParams::leaf_wise(8))
        .cache_size(32)
        .build()
        .expect("Invalid configuration");

    let model_leaf = GBDTModel::train(&dataset, ArrayView1::from(&labels[..]), None, config_leaf, 1)
        .expect("Training failed");

    let predictions = model_leaf.predict(&row_matrix, 1);

    let acc = compute_accuracy(predictions.as_slice(), &labels);
    println!("Leaf-wise: {} trees", model_leaf.forest().n_trees());
    println!("Accuracy: {:.2}%", acc * 100.0);

    // =========================================================================
    // Sample Weighting Example
    // =========================================================================
    println!("\n=== Training with Sample Weights ===\n");

    // Give higher weights to samples near decision boundary
    let weights: Vec<f32> = features
        .chunks(n_features)
        .map(|row| {
            let dx = row[0] - 5.0;
            let dy = row[1] - 5.0;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist >= 2.0 && dist <= 4.0 { 3.0 } else { 1.0 }
        })
        .collect();

    let config_weighted = GBDTConfig::builder()
        .objective(Objective::logistic())
        .metric(Metric::logloss())
        .n_trees(30)
        .learning_rate(0.1)
        .tree(TreeParams::depth_wise(3))
        .cache_size(32)
        .build()
        .expect("Invalid configuration");

    let model_weighted = GBDTModel::train(
        &dataset,
        ArrayView1::from(&labels[..]),
        Some(ArrayView1::from(&weights[..])),
        config_weighted,
        1,
    )
    .expect("Training failed");

    let predictions = model_weighted.predict(&row_matrix, 1);

    let acc = compute_accuracy(predictions.as_slice(), &labels);
    println!("Weighted training: {} trees", model_weighted.forest().n_trees());
    println!("Accuracy: {:.2}%", acc * 100.0);
    println!("\nNote: See train_imbalanced.rs for class imbalance handling.");
}

/// Compute classification accuracy.
fn compute_accuracy(probs: &[f32], labels: &[f32]) -> f32 {
    let correct: usize = probs
        .iter()
        .zip(labels.iter())
        .filter(|(prob, label)| {
            let pred = if **prob >= 0.5 { 1.0 } else { 0.0 };
            (pred - **label).abs() < 0.5
        })
        .count();
    correct as f32 / labels.len() as f32
}
