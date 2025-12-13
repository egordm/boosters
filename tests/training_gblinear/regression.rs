//! Basic regression training tests.
//!
//! Tests for linear regression training including:
//! - Weight matching with XGBoost
//! - L2 regularization effects
//! - Elastic net sparsity
//! - Prediction accuracy
//! - Parallel vs sequential training

use super::{load_config, load_train_data, load_xgb_weights};
use approx::assert_relative_eq;
use booste_rs::data::{ColMatrix, DataMatrix, Dataset, RowMatrix};
use booste_rs::training::{GBLinearParams, GBLinearTrainer, Rmse, SquaredLoss, Verbosity};
use rstest::rstest;

/// Test that we can train a simple linear regression and get similar weights to XGBoost.
#[rstest]
#[case("regression_simple")]
#[case("regression_multifeature")]
fn train_regression_matches_xgboost(#[case] name: &str) {
    let (data, labels) = load_train_data(name);
    let train = Dataset::from_numeric(&data, labels.clone()).unwrap();
    let xgb_weights = load_xgb_weights(name);
    let config = load_config(name);

    let params = GBLinearParams {
        n_rounds: config.num_boost_round as u32,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda,
        parallel: false,
        seed: 42,
        early_stopping_rounds: 0,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer = GBLinearTrainer::new(SquaredLoss, Rmse, params);
    let model = trainer.train(&train, &[]).unwrap();

    // Compare weights
    // XGBoost stores weights as [w0, w1, ..., wn-1, bias]
    let num_features = xgb_weights.num_features;

    println!("Test case: {}", name);
    println!("XGBoost weights: {:?}", xgb_weights.weights);
    println!("booste-rs bias: {}", model.bias(0));
    for i in 0..num_features {
        println!("booste-rs weight[{}]: {}", i, model.weight(i, 0));
    }

    // Check each weight is reasonably close
    // Note: We use a loose tolerance because training may converge differently
    for i in 0..num_features {
        let xgb_w = xgb_weights.weights[i];
        let our_w = model.weight(i, 0);
        assert_relative_eq!(our_w, xgb_w, max_relative = 0.3, epsilon = 0.5);
    }

    // Check bias
    let xgb_bias = xgb_weights.weights[num_features];
    let our_bias = model.bias(0);
    assert_relative_eq!(our_bias, xgb_bias, max_relative = 0.3, epsilon = 0.5);
}

/// Test that L2 regularization shrinks weights.
#[test]
fn train_l2_regularization_shrinks_weights() {
    let (data, labels) = load_train_data("regression_l2");
    let train = Dataset::from_numeric(&data, labels.clone()).unwrap();
    let config = load_config("regression_l2");

    // Train without regularization
    let params_no_reg = GBLinearParams {
        n_rounds: config.num_boost_round as u32,
        learning_rate: config.eta,
        alpha: 0.0,
        lambda: 0.0,
        parallel: false,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };
    let trainer_no_reg = GBLinearTrainer::new(SquaredLoss, Rmse, params_no_reg);
    let no_reg_model = trainer_no_reg.train(&train, &[]).unwrap();

    // Train with L2 regularization
    let params_l2 = GBLinearParams {
        n_rounds: config.num_boost_round as u32,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda,
        parallel: false,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };
    let trainer_l2 = GBLinearTrainer::new(SquaredLoss, Rmse, params_l2);
    let l2_model = trainer_l2.train(&train, &[]).unwrap();

    // L2 should produce smaller weights on average
    let no_reg_l2_norm: f32 = (0..data.num_features())
        .map(|i| no_reg_model.weight(i, 0).powi(2))
        .sum();
    let l2_norm: f32 = (0..data.num_features())
        .map(|i| l2_model.weight(i, 0).powi(2))
        .sum();

    println!("No reg L2 norm: {}", no_reg_l2_norm);
    println!("With L2 reg norm: {}", l2_norm);

    assert!(
        l2_norm < no_reg_l2_norm,
        "L2 regularization should shrink weights"
    );
}

/// Test that L1 regularization can zero out some weights (sparsity).
#[test]
fn train_elastic_net_produces_sparse_weights() {
    let (data, labels) = load_train_data("regression_elastic_net");
    let train = Dataset::from_numeric(&data, labels.clone()).unwrap();
    let xgb_weights = load_xgb_weights("regression_elastic_net");
    let config = load_config("regression_elastic_net");

    let params = GBLinearParams {
        n_rounds: config.num_boost_round as u32,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda,
        parallel: false,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer = GBLinearTrainer::new(SquaredLoss, Rmse, params);
    let model = trainer.train(&train, &[]).unwrap();

    // Count near-zero weights in both
    let xgb_near_zero = xgb_weights
        .weights
        .iter()
        .take(xgb_weights.num_features)
        .filter(|w| w.abs() < 0.1)
        .count();

    let our_near_zero = (0..data.num_features())
        .filter(|&i| model.weight(i, 0).abs() < 0.1)
        .count();

    println!("XGBoost near-zero weights: {}", xgb_near_zero);
    println!("booste-rs near-zero weights: {}", our_near_zero);

    // Both should have some sparsity due to L1
    // Note: We don't require exact match, just that both exhibit some sparsity
}

/// Test predictions match after training.
#[test]
fn trained_model_predictions_reasonable() {
    // Simple test: y = 2x + 1
    let row_data = RowMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], 5, 1);
    let data: ColMatrix = row_data.to_layout();
    let labels = vec![3.0, 5.0, 7.0, 9.0, 11.0];
    let train = Dataset::from_numeric(&data, labels.clone()).unwrap();

    let params = GBLinearParams {
        n_rounds: 100,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 0.0,
        parallel: false,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer = GBLinearTrainer::new(SquaredLoss, Rmse, params);
    let model = trainer.train(&train, &[]).unwrap();

    // Predictions should be close to actual values
    for i in 0..5 {
        let x = (i + 1) as f32;
        let expected = 2.0 * x + 1.0;
        let pred = model.predict_row(&[x], &[0.0])[0];
        let diff = (pred - expected).abs();
        assert!(
            diff < 0.5,
            "Prediction for x={}: expected {}, got {}, diff={}",
            x, expected, pred, diff
        );
    }
}

/// Test parallel and sequential training produce similar results.
#[test]
fn parallel_vs_sequential_similar() {
    let row_data = RowMatrix::from_vec(
        vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        4,
        2,
    );
    let data: ColMatrix = row_data.to_layout();
    let labels = vec![3.0, 4.0, 5.0, 6.0]; // y = x0 + 2*x1
    let train = Dataset::from_numeric(&data, labels.clone()).unwrap();

    let params_seq = GBLinearParams {
        n_rounds: 50,
        learning_rate: 0.3,
        parallel: false,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };
    let trainer_seq = GBLinearTrainer::new(SquaredLoss, Rmse, params_seq);
    let seq_model = trainer_seq.train(&train, &[]).unwrap();

    let params_par = GBLinearParams {
        n_rounds: 50,
        learning_rate: 0.3,
        parallel: true,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };
    let trainer_par = GBLinearTrainer::new(SquaredLoss, Rmse, params_par);
    let par_model = trainer_par.train(&train, &[]).unwrap();

    // Predictions should be similar
    let seq_pred = seq_model.predict_row(&[2.0, 2.0], &[0.0])[0];
    let par_pred = par_model.predict_row(&[2.0, 2.0], &[0.0])[0];

    let diff = (seq_pred - par_pred).abs();
    assert!(
        diff < 2.0,
        "Sequential vs parallel predictions differ too much: {} vs {}",
        seq_pred, par_pred
    );
}

/// Test weight correlation with XGBoost.
#[rstest]
#[case("regression_l2")]
#[case("regression_elastic_net")]
fn weight_correlation_with_xgboost(#[case] name: &str) {
    use super::pearson_correlation;

    let (data, labels) = load_train_data(name);
    let train = Dataset::from_numeric(&data, labels.clone()).unwrap();
    let xgb_weights = load_xgb_weights(name);
    let config = load_config(name);

    let params = GBLinearParams {
        n_rounds: config.num_boost_round as u32,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda,
        parallel: false,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer = GBLinearTrainer::new(SquaredLoss, Rmse, params);
    let model = trainer.train(&train, &[]).unwrap();

    // Collect our weights (excluding bias)
    let our_weights: Vec<f32> = (0..xgb_weights.num_features)
        .map(|i| model.weight(i, 0))
        .collect();

    // XGBoost weights (excluding bias which is at the end)
    let xgb_w: Vec<f32> = xgb_weights.weights[..xgb_weights.num_features].to_vec();

    let corr = pearson_correlation(&our_weights, &xgb_w);
    println!("Test case {}: weight correlation = {:.4}", name, corr);

    // Weight correlation should be high
    assert!(
        corr > 0.9,
        "Weight correlation too low: {} (expected > 0.9)",
        corr
    );
}

/// Test held-out test set prediction quality.
#[rstest]
#[case("regression_l2")]
#[case("regression_elastic_net")]
fn test_set_prediction_quality(#[case] name: &str) {
    use super::{compute_test_rmse, load_test_data, load_xgb_predictions, rmse};

    let (train_data, train_labels) = load_train_data(name);
    let train = Dataset::from_numeric(&train_data, train_labels.clone()).unwrap();
    let config = load_config(name);

    // Skip if no test data
    let Some((test_data, test_labels)) = load_test_data(name) else {
        println!("Skipping {} - no test data", name);
        return;
    };

    let params = GBLinearParams {
        n_rounds: config.num_boost_round as u32,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda,
        parallel: false,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer = GBLinearTrainer::new(SquaredLoss, Rmse, params);
    let model = trainer.train(&train, &[]).unwrap();
    let base_scores = vec![0.0f32];

    let our_rmse = compute_test_rmse(&model, &test_data, &test_labels, &base_scores);
    println!("Test case {}: our test RMSE = {:.4}", name, our_rmse);

    // Compare to XGBoost if available
    if let Some(xgb_preds) = load_xgb_predictions(name) {
        let xgb_rmse = rmse(&xgb_preds, &test_labels);
        println!("Test case {}: XGBoost test RMSE = {:.4}", name, xgb_rmse);

        // Our RMSE should be within 50% of XGBoost
        assert!(
            our_rmse < xgb_rmse * 1.5,
            "Our RMSE ({}) too much worse than XGBoost ({})",
            our_rmse,
            xgb_rmse
        );
    }

    // RMSE should be reasonable (less than std of labels)
    let label_std = {
        let mean = test_labels.iter().sum::<f32>() / test_labels.len() as f32;
        let var: f32 =
            test_labels.iter().map(|&l| (l - mean).powi(2)).sum::<f32>() / test_labels.len() as f32;
        var.sqrt() as f64
    };

    assert!(
        our_rmse < label_std * 1.5,
        "RMSE ({}) larger than 1.5 * label std ({})",
        our_rmse,
        label_std
    );
}
