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
use boosters::training::{Accuracy, GBDTParams, GBDTTrainer, GrowthStrategy, LogLoss, LogisticLoss, MetricFn};
use boosters::Parallelism;

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

    let params_depth = GBDTParams {
        n_trees: 30,
        learning_rate: 0.1,
        growth_strategy: GrowthStrategy::DepthWise { max_depth: 3 },
        cache_size: 32,
        ..Default::default()
    };

    let trainer_depth = GBDTTrainer::new(LogisticLoss, LogLoss, params_depth);
    let forest_depth = trainer_depth
        .train(&dataset, &labels, &[], &[], Parallelism::SEQUENTIAL)
        .unwrap();

    // Predict: apply sigmoid to convert logits to probabilities
    let predictions: Vec<f32> = features
        .chunks(n_features)
        .map(|row| {
            let logit = forest_depth.predict_row(row)[0];
            1.0 / (1.0 + (-logit).exp())
        })
        .collect();

    let acc = Accuracy::default().compute(n_samples, 1, &predictions, &labels, &[]);
    println!("Depth-wise: {} trees", forest_depth.n_trees());
    println!("Accuracy: {:.2}%", acc * 100.0);

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

    let trainer_leaf = GBDTTrainer::new(LogisticLoss, LogLoss, params_leaf);
    let forest_leaf = trainer_leaf
        .train(&dataset, &labels, &[], &[], Parallelism::SEQUENTIAL)
        .unwrap();

    let predictions: Vec<f32> = features
        .chunks(n_features)
        .map(|row| {
            let logit = forest_leaf.predict_row(row)[0];
            1.0 / (1.0 + (-logit).exp())
        })
        .collect();

    let acc = Accuracy::default().compute(n_samples, 1, &predictions, &labels, &[]);
    println!("Leaf-wise: {} trees", forest_leaf.n_trees());
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

    let forest_weighted = trainer_depth
        .train(&dataset, &labels, &weights, &[], Parallelism::SEQUENTIAL)
        .unwrap();

    let predictions: Vec<f32> = features
        .chunks(n_features)
        .map(|row| {
            let logit = forest_weighted.predict_row(row)[0];
            1.0 / (1.0 + (-logit).exp())
        })
        .collect();

    let acc = Accuracy::default().compute(n_samples, 1, &predictions, &labels, &[]);
    println!("Weighted training: {} trees", forest_weighted.n_trees());
    println!("Accuracy: {:.2}%", acc * 100.0);
    println!("\nNote: See train_imbalanced.rs for class imbalance handling.");
}
