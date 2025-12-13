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

use booste_rs::data::binned::BinnedDatasetBuilder;
use booste_rs::data::{ColMatrix, DenseMatrix, RowMajor};
use booste_rs::training::{GBDTParams, GBDTTrainer, GrowthStrategy, LogisticLoss};

fn main() {
    // Generate synthetic binary classification data
    // Two clusters: class 0 centered at (2, 2), class 1 centered at (8, 8)
    let n_samples = 400;
    let n_features = 4;

    let mut features = Vec::with_capacity(n_samples * n_features);
    let mut labels = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let class = (i % 2) as f32;
        let offset = if class == 0.0 { 2.0 } else { 8.0 };

        // Features with some randomness
        let noise1 = ((i * 17) % 100) as f32 / 50.0 - 1.0;
        let noise2 = ((i * 23) % 100) as f32 / 50.0 - 1.0;
        let noise3 = ((i * 31) % 100) as f32 / 50.0 - 1.0;
        let noise4 = ((i * 37) % 100) as f32 / 50.0 - 1.0;

        features.push(offset + noise1);
        features.push(offset + noise2);
        features.push(noise3 * 2.0); // Random feature
        features.push(noise4 * 2.0); // Random feature

        labels.push(class);
    }

    // Keep a copy of features for weight computation
    let features_for_weights = features.clone();

    // Create row-major matrix and convert to column-major for training
    let row_matrix: DenseMatrix<f32, RowMajor> =
        DenseMatrix::from_vec(features, n_samples, n_features);
    let col_matrix: ColMatrix<f32> = row_matrix.to_layout();

    // Create binned dataset
    let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 256)
        .build()
        .expect("Failed to build binned dataset");

    // =========================================================================
    // Train with depth-wise growth (XGBoost style)
    // =========================================================================
    println!("=== Depth-wise Growth (XGBoost style) ===\n");

    let params_depth = GBDTParams {
        n_trees: 30,
        learning_rate: 0.1,
        growth_strategy: GrowthStrategy::DepthWise { max_depth: 3 },
        cache_size: 32,
        ..Default::default()
    };

    let trainer_depth = GBDTTrainer::new(LogisticLoss, params_depth);
    let forest_depth = trainer_depth.train(&dataset, &labels, &[]).unwrap();

    // Evaluate depth-wise
    let preds_depth = predict_all(&dataset, &forest_depth);
    let acc_depth = accuracy(&preds_depth, &labels);

    println!("Depth-wise: {} trees", forest_depth.n_trees());
    println!("Accuracy: {:.2}%", acc_depth * 100.0);

    // =========================================================================
    // Train with leaf-wise growth (LightGBM style)
    // =========================================================================
    println!("\n=== Leaf-wise Growth (LightGBM style) ===\n");

    let params_leaf = GBDTParams {
        n_trees: 30,
        learning_rate: 0.1,
        growth_strategy: GrowthStrategy::LeafWise { max_leaves: 8 },
        cache_size: 32,
        ..Default::default()
    };

    let trainer_leaf = GBDTTrainer::new(LogisticLoss, params_leaf);
    let forest_leaf = trainer_leaf.train(&dataset, &labels, &[]).unwrap();

    // Evaluate leaf-wise
    let preds_leaf = predict_all(&dataset, &forest_leaf);
    let acc_leaf = accuracy(&preds_leaf, &labels);

    println!("Leaf-wise: {} trees", forest_leaf.n_trees());
    println!("Accuracy: {:.2}%", acc_leaf * 100.0);

    // =========================================================================
    // Sample Weighting Example
    // =========================================================================
    println!("\n=== Training with Sample Weights ===\n");

    // Example: Give higher weights to samples near decision boundary
    let weights: Vec<f32> = features_for_weights
        .chunks(n_features)
        .map(|row| {
            // Distance from center (5, 5)
            let dx = row[0] - 5.0;
            let dy = row[1] - 5.0;
            let dist = (dx * dx + dy * dy).sqrt();
            // Higher weight for samples near decision boundary (dist ~2-4)
            if dist >= 2.0 && dist <= 4.0 {
                3.0 // Emphasize boundary samples
            } else {
                1.0
            }
        })
        .collect();

    let forest_weighted = trainer_depth
        .train(&dataset, &labels, &weights)
        .unwrap();
    let preds_weighted = predict_all(&dataset, &forest_weighted);
    let acc_weighted = accuracy(&preds_weighted, &labels);

    println!("Weighted training: {} trees", forest_weighted.n_trees());
    println!("Accuracy: {:.2}%", acc_weighted * 100.0);
    println!("\nNote: See train_imbalanced.rs for class imbalance handling.");
}

/// Predict for all rows using binned data.
fn predict_all(
    dataset: &booste_rs::data::binned::BinnedDataset,
    forest: &booste_rs::training::gbdt::tree::Forest,
) -> Vec<f32> {
    let base = forest.base_scores()[0];
    let mut preds = Vec::with_capacity(dataset.n_rows());

    for row_idx in 0..dataset.n_rows() {
        if let Some(row_view) = dataset.row_view(row_idx) {
            let tree_sum: f32 = forest.trees().iter().map(|t| t.predict(&row_view)).sum();
            preds.push(base + tree_sum);
        }
    }
    preds
}

/// Compute classification accuracy.
///
/// Predictions are raw scores; we apply sigmoid and threshold at 0.5.
fn accuracy(predictions: &[f32], labels: &[f32]) -> f32 {
    let correct: usize = predictions
        .iter()
        .zip(labels.iter())
        .filter(|(pred, label)| {
            let prob = 1.0 / (1.0 + (-**pred).exp()); // sigmoid
            let predicted_class = if prob >= 0.5 { 1.0 } else { 0.0 };
            (predicted_class - **label).abs() < 0.5
        })
        .count();
    correct as f32 / predictions.len() as f32
}
