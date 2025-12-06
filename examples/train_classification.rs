//! GBTree binary classification training example.
//!
//! This example demonstrates training a gradient boosted tree model for
//! binary classification using logistic loss.
//!
//! Run with:
//! ```bash
//! cargo run --example train_classification
//! ```

use booste_rs::data::{ColMatrix, RowMatrix};
use booste_rs::predict::{Predictor, StandardTraversal};
use booste_rs::training::gbtree::ExactQuantileCuts;
use booste_rs::training::{
    DepthWisePolicy, GBTreeTrainer, LeafWisePolicy, LogisticLoss, Quantizer, TrainerParams,
    TreeParams, Verbosity,
};

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

    // Create matrix
    let row_matrix = RowMatrix::from_vec(features, n_samples, n_features);
    let col_matrix: ColMatrix<f32> = row_matrix.to_layout();

    // Quantize
    let cut_finder = ExactQuantileCuts::default();
    let quantizer = Quantizer::from_data(&col_matrix, &cut_finder, 256);
    let cuts = quantizer.cuts().clone();
    let quantized = quantizer.quantize::<_, u8>(&col_matrix);

    // Configure training - shallow trees for classification
    let tree_params = TreeParams {
        max_depth: 3,
        max_leaves: 8,
        learning_rate: 0.1,
        ..Default::default()
    };

    let params = TrainerParams {
        num_rounds: 30,
        tree_params,
        verbosity: Verbosity::Info,
        ..Default::default()
    };

    // Train with logistic loss (binary classification)
    let mut trainer = GBTreeTrainer::new(Box::new(LogisticLoss), params);

    // Demonstrate depth-wise growth strategy
    println!("=== Depth-wise Growth (XGBoost style) ===\n");
    let depth_policy = DepthWisePolicy { max_depth: 3 };
    let forest_depth = trainer.train(depth_policy, &quantized, &labels, &cuts, &[]);

    // Evaluate depth-wise
    let predictor_depth = Predictor::<StandardTraversal>::new(&forest_depth);
    let preds_depth = predictor_depth.predict(&row_matrix).into_vec();
    let acc_depth = accuracy(&preds_depth, &labels);

    println!("\nDepth-wise: {} trees", forest_depth.num_trees());
    println!("Accuracy: {:.2}%", acc_depth * 100.0);

    // Train with leaf-wise growth
    let tree_params_leaf = TreeParams {
        max_depth: 0, // No depth limit for leaf-wise
        max_leaves: 8,
        learning_rate: 0.1,
        ..Default::default()
    };

    let params_leaf = TrainerParams {
        num_rounds: 30,
        tree_params: tree_params_leaf,
        verbosity: Verbosity::Info,
        ..Default::default()
    };

    let mut trainer_leaf = GBTreeTrainer::new(Box::new(LogisticLoss), params_leaf);

    println!("\n=== Leaf-wise Growth (LightGBM style) ===\n");
    let leaf_policy = LeafWisePolicy { max_leaves: 8 };
    let forest_leaf = trainer_leaf.train(leaf_policy, &quantized, &labels, &cuts, &[]);

    // Evaluate leaf-wise
    let predictor_leaf = Predictor::<StandardTraversal>::new(&forest_leaf);
    let preds_leaf = predictor_leaf.predict(&row_matrix).into_vec();
    let acc_leaf = accuracy(&preds_leaf, &labels);

    println!("\nLeaf-wise: {} trees", forest_leaf.num_trees());
    println!("Accuracy: {:.2}%", acc_leaf * 100.0);
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
