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

use boosters::data::Dataset;
use boosters::training::GrowthStrategy;
use boosters::{GBDTConfig, GBDTModel, Metric, Objective};
use ndarray::{Array1, Array2};

fn main() {
    // =========================================================================
    // Generate synthetic binary classification data
    // =========================================================================
    // Two clusters: class 0 centered at (2, 2), class 1 centered at (8, 8)
    let n_samples = 400;
    let n_features = 4;

    // Feature-major data [n_features, n_samples]
    let mut features = Array2::<f32>::zeros((n_features, n_samples));
    let mut labels = Array1::<f32>::zeros(n_samples);

    for i in 0..n_samples {
        let class = (i % 2) as f32;
        let offset = if class == 0.0 { 2.0 } else { 8.0 };

        let noise1 = ((i * 17) % 100) as f32 / 50.0 - 1.0;
        let noise2 = ((i * 23) % 100) as f32 / 50.0 - 1.0;
        let noise3 = ((i * 31) % 100) as f32 / 50.0 - 1.0;
        let noise4 = ((i * 37) % 100) as f32 / 50.0 - 1.0;

        features[(0, i)] = offset + noise1;
        features[(1, i)] = offset + noise2;
        features[(2, i)] = noise3 * 2.0;
        features[(3, i)] = noise4 * 2.0;

        labels[i] = class;
    }

    // Wrap labels in targets array (shape [n_outputs=1, n_samples])
    let targets_2d = labels.clone().insert_axis(ndarray::Axis(0));
    let dataset = Dataset::from_array(features.view(), Some(targets_2d.clone()), None);

    // =========================================================================
    // Train with depth-wise growth (XGBoost style)
    // =========================================================================
    println!("=== Depth-wise Growth (XGBoost style) ===\n");

    let config_depth = GBDTConfig::builder()
        .objective(Objective::LogisticLoss)
        .metric(Metric::LogLoss)
        .n_trees(30)
        .learning_rate(0.1)
        .growth_strategy(GrowthStrategy::DepthWise { max_depth: 3 })
        .cache_size(32)
        .build()
        .expect("Invalid configuration");

    let model_depth = GBDTModel::train(&dataset, None, config_depth, 1).expect("Training failed");

    let predictions = model_depth.predict(&dataset, 1);

    let acc = compute_accuracy(predictions.as_slice().unwrap(), labels.as_slice().unwrap());
    println!("Depth-wise: {} trees", model_depth.forest().n_trees());
    println!("Accuracy: {:.2}%", acc * 100.0);

    // =========================================================================
    // Train with leaf-wise growth (LightGBM style)
    // =========================================================================
    println!("\n=== Leaf-wise Growth (LightGBM style) ===\n");

    let config_leaf = GBDTConfig::builder()
        .objective(Objective::LogisticLoss)
        .metric(Metric::LogLoss)
        .n_trees(30)
        .learning_rate(0.1)
        .growth_strategy(GrowthStrategy::LeafWise { max_leaves: 8 })
        .cache_size(32)
        .build()
        .expect("Invalid configuration");

    let model_leaf = GBDTModel::train(&dataset, None, config_leaf, 1).expect("Training failed");

    let predictions = model_leaf.predict(&dataset, 1);

    let acc = compute_accuracy(predictions.as_slice().unwrap(), labels.as_slice().unwrap());
    println!("Leaf-wise: {} trees", model_leaf.forest().n_trees());
    println!("Accuracy: {:.2}%", acc * 100.0);

    // =========================================================================
    // Sample Weighting Example
    // =========================================================================
    println!("\n=== Training with Sample Weights ===\n");

    // Give higher weights to samples near decision boundary
    let weights: Vec<f32> = (0..n_samples)
        .map(|i| {
            let dx = features[(0, i)] - 5.0;
            let dy = features[(1, i)] - 5.0;
            let dist = (dx * dx + dy * dy).sqrt();
            if (2.0..=4.0).contains(&dist) {
                3.0
            } else {
                1.0
            }
        })
        .collect();

    let config_weighted = GBDTConfig::builder()
        .objective(Objective::LogisticLoss)
        .metric(Metric::LogLoss)
        .n_trees(30)
        .learning_rate(0.1)
        .growth_strategy(GrowthStrategy::DepthWise { max_depth: 3 })
        .cache_size(32)
        .build()
        .expect("Invalid configuration");

    let dataset_weighted = Dataset::from_array(
        features.view(),
        Some(targets_2d),
        Some(Array1::from_vec(weights)),
    );
    let model_weighted =
        GBDTModel::train(&dataset_weighted, None, config_weighted, 1).expect("Training failed");

    let predictions = model_weighted.predict(&dataset_weighted, 1);

    let acc = compute_accuracy(predictions.as_slice().unwrap(), labels.as_slice().unwrap());
    println!(
        "Weighted training: {} trees",
        model_weighted.forest().n_trees()
    );
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
