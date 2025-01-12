//! Basic regression training tests.
//!
//! Tests for linear regression training including:
//! - Weight matching with XGBoost
//! - L2 regularization effects
//! - Elastic net sparsity
//! - Prediction accuracy
//! - Parallel vs sequential training

use super::{load_config, load_train_data, load_xgb_weights, make_dataset};
use approx::assert_relative_eq;
use boosters::data::transpose_to_c_order;
use boosters::data::{FeaturesView, TargetsView, WeightsView};
use boosters::training::{GBLinearParams, GBLinearTrainer, Rmse, SquaredLoss, Verbosity};
use ndarray::Array2;
use rstest::rstest;

/// Test that we can train a simple linear regression and get similar weights to XGBoost.
#[rstest]
#[case("regression_simple")]
#[case("regression_multifeature")]
fn train_regression_matches_xgboost(#[case] name: &str) {
    let (data, labels) = load_train_data(name);
    let (train, targets) = make_dataset(&data, &labels);
    let targets_view = TargetsView::new(targets.view());
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
    let model = trainer
        .train(&train, targets_view, WeightsView::None, &[])
        .unwrap();

    // Compare weights
    // XGBoost stores weights as [w0, w1, ..., wn-1, bias]
    let num_features = xgb_weights.num_features;

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
    let n_features = data.nrows();
    let (train, targets) = make_dataset(&data, &labels);
    let targets_view = TargetsView::new(targets.view());
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
    let no_reg_model = trainer_no_reg
        .train(&train, targets_view.clone(), WeightsView::None, &[])
        .unwrap();

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
    let l2_model = trainer_l2
        .train(&train, targets_view, WeightsView::None, &[])
        .unwrap();

    // L2 should produce smaller weights on average
    let no_reg_l2_norm: f32 = (0..n_features)
        .map(|i| no_reg_model.weight(i, 0).powi(2))
        .sum();
    let l2_norm: f32 = (0..n_features).map(|i| l2_model.weight(i, 0).powi(2)).sum();

    assert!(
        l2_norm < no_reg_l2_norm,
        "L2 regularization should shrink weights"
    );
}

/// Test predictions match after training.
#[test]
fn trained_model_predictions_reasonable() {
    // Simple test: y = 2x + 1
    // Feature-major: [n_features=1, n_samples=5]
    let data = Array2::from_shape_vec((1, 5), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let labels = vec![3.0, 5.0, 7.0, 9.0, 11.0];
    let (train, targets) = make_dataset(&data, &labels);
    let targets_view = TargetsView::new(targets.view());

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
    let model = trainer
        .train(&train, targets_view, WeightsView::None, &[])
        .unwrap();

    // Predictions should be close to actual values
    for i in 0..5 {
        let x = (i + 1) as f32;
        let expected = 2.0 * x + 1.0;
        let mut output = [0.0f32; 1];
        model.predict_row_into(&[x], &mut output);
        let pred = output[0];
        let diff = (pred - expected).abs();
        assert!(
            diff < 0.5,
            "Prediction for x={}: expected {}, got {}, diff={}",
            x,
            expected,
            pred,
            diff
        );
    }
}

/// Test parallel and sequential training produce similar results.
#[test]
fn parallel_vs_sequential_similar() {
    // Feature-major: [n_features=2, n_samples=4]
    // Converting from row-major: [[1,1], [2,1], [1,2], [2,2]] -> feature 0: [1,2,1,2], feature 1: [1,1,2,2]
    let data = Array2::from_shape_vec(
        (2, 4),
        vec![
            1.0, 2.0, 1.0, 2.0, // feature 0
            1.0, 1.0, 2.0, 2.0, // feature 1
        ],
    )
    .unwrap();
    let labels = vec![3.0, 4.0, 5.0, 6.0]; // y = x0 + 2*x1
    let (train, targets) = make_dataset(&data, &labels);
    let targets_view = TargetsView::new(targets.view());

    let params_seq = GBLinearParams {
        n_rounds: 50,
        learning_rate: 0.3,
        parallel: false,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };
    let trainer_seq = GBLinearTrainer::new(SquaredLoss, Rmse, params_seq);
    let seq_model = trainer_seq
        .train(&train, targets_view.clone(), WeightsView::None, &[])
        .unwrap();

    let params_par = GBLinearParams {
        n_rounds: 50,
        learning_rate: 0.3,
        parallel: true,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };
    let trainer_par = GBLinearTrainer::new(SquaredLoss, Rmse, params_par);
    let par_model = trainer_par
        .train(&train, targets_view, WeightsView::None, &[])
        .unwrap();

    // Predictions should be similar
    let mut seq_output = [0.0f32; 1];
    let mut par_output = [0.0f32; 1];
    seq_model.predict_row_into(&[2.0, 2.0], &mut seq_output);
    par_model.predict_row_into(&[2.0, 2.0], &mut par_output);
    let seq_pred = seq_output[0];
    let par_pred = par_output[0];

    let diff = (seq_pred - par_pred).abs();
    assert!(
        diff < 2.0,
        "Sequential vs parallel predictions differ too much: {} vs {}",
        seq_pred,
        par_pred
    );
}

/// Test weight correlation with XGBoost.
#[rstest]
#[case("regression_l2")]
#[case("regression_elastic_net")]
fn weight_correlation_with_xgboost(#[case] name: &str) {
    use super::pearson_correlation;

    let (data, labels) = load_train_data(name);
    let (train, targets) = make_dataset(&data, &labels);
    let targets_view = TargetsView::new(targets.view());
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
    let model = trainer
        .train(&train, targets_view, WeightsView::None, &[])
        .unwrap();

    // Collect our weights (excluding bias)
    let our_weights: Vec<f32> = (0..xgb_weights.num_features)
        .map(|i| model.weight(i, 0))
        .collect();

    // XGBoost weights (excluding bias which is at the end)
    let xgb_w: Vec<f32> = xgb_weights.weights[..xgb_weights.num_features].to_vec();

    let corr = pearson_correlation(&our_weights, &xgb_w);

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
    use super::{load_test_data, load_xgb_predictions, rmse};

    let (train_data, train_labels) = load_train_data(name);
    let (train, targets) = make_dataset(&train_data, &train_labels);
    let targets_view = TargetsView::new(targets.view());
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
    let model = trainer
        .train(&train, targets_view, WeightsView::None, &[])
        .unwrap();

    use boosters::training::MetricFn;
    // test_data is sample-major [n_samples, n_features], need feature-major [n_features, n_samples]
    let test_features = transpose_to_c_order(test_data.view());
    let test_view = FeaturesView::from_array(test_features.view());
    let output = model.predict(test_view);
    // output is [n_groups, n_samples], metrics expect [n_outputs, n_samples]
    let targets_2d = Array2::from_shape_vec((1, test_labels.len()), test_labels.clone()).unwrap();
    let test_targets = TargetsView::new(targets_2d.view());
    let our_rmse = Rmse.compute(output.view(), test_targets, WeightsView::None);

    // Compare to XGBoost if available
    if let Some(xgb_preds) = load_xgb_predictions(name) {
        let xgb_rmse = rmse(&xgb_preds, &test_labels);

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
