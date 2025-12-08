//! GBTree binary classification training example.
//!
//! This example demonstrates training a gradient boosted tree model for
//! binary classification using logistic loss.
//!
//! ## Features Shown
//!
//! - Binary classification with `LossFunction::Logistic`
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

use booste_rs::data::{ColMatrix, RowMatrix};
use booste_rs::predict::{Predictor, StandardTraversal};
use booste_rs::training::{GBTreeTrainer, GrowthMode, LossFunction, Verbosity};

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

    // Keep a copy of features for weight computation (before moving into matrix)
    let features_for_weights = features.clone();

    // Create matrix
    let row_matrix = RowMatrix::from_vec(features, n_samples, n_features);
    let col_matrix: ColMatrix<f32> = row_matrix.to_layout();

    // Train with depth-wise growth (XGBoost style)
    println!("=== Depth-wise Growth (XGBoost style) ===\n");

    let trainer_depth = GBTreeTrainer::builder()
        .loss(LossFunction::Logistic)
        .num_rounds(30u32)
        .max_depth(3u8)
        .learning_rate(0.1f32)
        .verbosity(Verbosity::Info)
        .build()
        .unwrap();

    let forest_depth = trainer_depth.train(&col_matrix, &labels, None, &[]);

    // Evaluate depth-wise
    let predictor_depth = Predictor::<StandardTraversal>::new(&forest_depth);
    let preds_depth = predictor_depth.predict(&row_matrix).into_vec();
    let acc_depth = accuracy(&preds_depth, &labels);

    println!("\nDepth-wise: {} trees", forest_depth.num_trees());
    println!("Accuracy: {:.2}%", acc_depth * 100.0);

    // Train with leaf-wise growth (LightGBM style)
    println!("\n=== Leaf-wise Growth (LightGBM style) ===\n");

    let trainer_leaf = GBTreeTrainer::builder()
        .loss(LossFunction::Logistic)
        .num_rounds(30u32)
        .growth_mode(GrowthMode::LeafWise) // Leaf-wise growth (LightGBM style)
        .max_leaves(8u32)
        .learning_rate(0.1f32)
        .verbosity(Verbosity::Info)
        .build()
        .unwrap();

    let forest_leaf = trainer_leaf.train(&col_matrix, &labels, None, &[]);

    // Evaluate leaf-wise
    let predictor_leaf = Predictor::<StandardTraversal>::new(&forest_leaf);
    let preds_leaf = predictor_leaf.predict(&row_matrix).into_vec();
    let acc_leaf = accuracy(&preds_leaf, &labels);

    println!("\nLeaf-wise: {} trees", forest_leaf.num_trees());
    println!("Accuracy: {:.2}%", acc_leaf * 100.0);

    // =========================================================================
    // Sample Weighting Example
    // =========================================================================

    println!("\n=== Training with Sample Weights ===\n");

    // Example: Give higher weights to samples near decision boundary
    // This can improve accuracy in the challenging region
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

    let forest_weighted = trainer_depth.train(&col_matrix, &labels, Some(&weights), &[]);
    let predictor_weighted = Predictor::<StandardTraversal>::new(&forest_weighted);
    let preds_weighted = predictor_weighted.predict(&row_matrix).into_vec();
    let acc_weighted = accuracy(&preds_weighted, &labels);

    println!("Weighted training: {} trees", forest_weighted.num_trees());
    println!("Accuracy: {:.2}%", acc_weighted * 100.0);
    println!("\nNote: See train_imbalanced.rs for class imbalance handling.");
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
