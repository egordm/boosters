//! XGBoost inference tests: model loading, parsing, and prediction.
//!
//! These tests verify that booste-rs can correctly:
//! 1. Parse XGBoost JSON model files
//! 2. Convert models to internal representation
//! 3. Produce predictions matching Python XGBoost
//!
//! Test cases organized by booster type:
//! - `xgboost/gbtree/{name}.*` - GBTree models
//! - `xgboost/gblinear/{name}.*` - GBLinear models
//! - `xgboost/dart/{name}.*` - DART models

#![cfg(feature = "xgboost-compat")]

mod common;

use std::fs::File;
use std::path::PathBuf;

use booste_rs::compat::XgbModel;

use common::{TestExpected, TestInput, DEFAULT_TOLERANCE_F64};

// =============================================================================
// Test Case Loading
// =============================================================================

fn gbtree_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test-cases/xgboost/gbtree")
}

fn gblinear_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test-cases/xgboost/gblinear")
}

fn dart_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test-cases/xgboost/dart")
}

fn load_from_dir(dir: &std::path::Path, name: &str) -> (XgbModel, TestInput, TestExpected) {
    let model: XgbModel = serde_json::from_reader(
        File::open(dir.join(format!("{name}.model.json"))).expect("model file"),
    )
    .expect("parse model");

    let input: TestInput = serde_json::from_reader(
        File::open(dir.join(format!("{name}.input.json"))).expect("input file"),
    )
    .expect("parse input");

    let expected: TestExpected = serde_json::from_reader(
        File::open(dir.join(format!("{name}.expected.json"))).expect("expected file"),
    )
    .expect("parse expected");

    (model, input, expected)
}

fn load_gbtree(name: &str) -> (XgbModel, TestInput, TestExpected) {
    load_from_dir(&gbtree_dir(), name)
}

fn load_gblinear(name: &str) -> (XgbModel, TestInput, TestExpected) {
    load_from_dir(&gblinear_dir(), name)
}

fn load_dart(name: &str) -> (XgbModel, TestInput, TestExpected) {
    load_from_dir(&dart_dir(), name)
}

// =============================================================================
// Helper functions
// =============================================================================

/// Convert probability-space base_score to margin space based on objective.
fn prob_to_margin(base_score: f32, objective: &str) -> f32 {
    match objective {
        "binary:logistic" | "reg:logistic" => {
            let p = base_score.clamp(1e-7, 1.0 - 1e-7);
            (p / (1.0 - p)).ln()
        }
        "reg:gamma" | "reg:tweedie" => base_score.max(1e-7).ln(),
        _ => base_score,
    }
}

/// Assert predictions match expected values within tolerance.
fn assert_preds_match(actual: &[f32], expected: &[f64], tolerance: f64, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: length mismatch - got {}, expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (*a as f64 - *e).abs();
        // Use both absolute and relative tolerance for f32 precision
        let rel_tol = e.abs() * 1e-6;
        assert!(
            diff < tolerance || diff < rel_tol,
            "{context}[{i}]: got {a}, expected {e}, diff {diff}"
        );
    }
}

// =============================================================================
// GBTree Parsing Tests
// =============================================================================

#[test]
fn parse_regression_model() {
    let (model, input, _) = load_gbtree("gbtree_regression");
    assert_eq!(input.num_features, 5);
    assert_eq!(input.num_rows, 10);
    assert_eq!(model.learner.learner_model_param.num_class, 0);
}

#[test]
fn parse_binary_logistic_model() {
    let (model, input, _) = load_gbtree("gbtree_binary_logistic");
    assert_eq!(input.num_features, 4);
    assert_eq!(model.learner.learner_model_param.num_class, 0);
}

#[test]
fn parse_multiclass_model() {
    let (model, input, _) = load_gbtree("gbtree_multiclass");
    assert_eq!(input.num_features, 4);
    assert_eq!(model.learner.learner_model_param.num_class, 3);
}

#[test]
fn parse_model_with_missing_values() {
    let (model, input, _) = load_gbtree("gbtree_regression_missing");
    assert_eq!(input.num_features, 4);
    // Check that missing values are present in input (None = NaN)
    assert!(input.features[0][0].is_none());
    assert!(input.features[2][1].is_none());
    assert!(input.features[5].iter().all(|x| x.is_none()));
    assert!(model.to_forest().is_ok());
}

#[test]
fn parse_deep_trees_model() {
    let (model, input, _) = load_gbtree("gbtree_deep_trees");
    assert_eq!(input.num_features, 8);
    assert!(model.to_forest().is_ok());
}

#[test]
fn parse_single_tree_model() {
    let (model, input, _) = load_gbtree("gbtree_single_tree");
    assert_eq!(input.num_features, 3);
    let forest = model.to_forest().expect("conversion failed");
    assert_eq!(forest.num_trees(), 1);
}

#[test]
fn parse_many_trees_model() {
    let (model, input, _) = load_gbtree("gbtree_many_trees");
    assert_eq!(input.num_features, 5);
    let forest = model.to_forest().expect("conversion failed");
    assert_eq!(forest.num_trees(), 50);
}

#[test]
fn parse_wide_features_model() {
    let (model, input, _) = load_gbtree("gbtree_wide_features");
    assert_eq!(input.num_features, 100);
    assert!(model.to_forest().is_ok());
}

// =============================================================================
// GBTree Conversion Tests
// =============================================================================

#[test]
fn convert_regression_model() {
    let (model, _, _) = load_gbtree("gbtree_regression");
    let forest = model.to_forest().expect("conversion failed");
    assert_eq!(forest.num_groups(), 1);
    assert!(forest.num_trees() > 0);
}

#[test]
fn convert_binary_logistic_model() {
    let (model, _, _) = load_gbtree("gbtree_binary_logistic");
    let forest = model.to_forest().expect("conversion failed");
    assert_eq!(forest.num_groups(), 1);
}

#[test]
fn convert_multiclass_model() {
    let (model, _, _) = load_gbtree("gbtree_multiclass");
    let forest = model.to_forest().expect("conversion failed");
    assert_eq!(forest.num_groups(), 3);
}

// =============================================================================
// GBTree Prediction Tests
// =============================================================================

#[test]
fn predict_regression() {
    let (model, input, expected) = load_gbtree("gbtree_regression");
    let forest = model.to_forest().expect("conversion failed");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = forest.predict_row(features);
        assert_preds_match(&pred, &[expected_preds[i]], DEFAULT_TOLERANCE_F64, &format!("row {i}"));
    }
}

#[test]
fn predict_binary_logistic() {
    let (model, input, expected) = load_gbtree("gbtree_binary_logistic");
    let forest = model.to_forest().expect("conversion failed");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = forest.predict_row(features);
        assert_preds_match(&pred, &[expected_preds[i]], DEFAULT_TOLERANCE_F64, &format!("row {i}"));
    }
}

#[test]
fn predict_multiclass() {
    let (model, input, expected) = load_gbtree("gbtree_multiclass");
    let forest = model.to_forest().expect("conversion failed");

    let expected_preds = expected.as_nested();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = forest.predict_row(features);
        assert_eq!(pred.len(), 3, "expected 3 classes");
        assert_preds_match(&pred, &expected_preds[i], DEFAULT_TOLERANCE_F64, &format!("row {i}"));
    }
}

#[test]
fn predict_with_missing_values() {
    let (model, input, expected) = load_gbtree("gbtree_regression_missing");
    let forest = model.to_forest().expect("conversion failed");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = forest.predict_row(features);
        assert_preds_match(&pred, &[expected_preds[i]], DEFAULT_TOLERANCE_F64, &format!("row {i}"));
    }
}

#[test]
fn predict_deep_trees() {
    let (model, input, expected) = load_gbtree("gbtree_deep_trees");
    let forest = model.to_forest().expect("conversion failed");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = forest.predict_row(features);
        assert_preds_match(&pred, &[expected_preds[i]], DEFAULT_TOLERANCE_F64, &format!("row {i}"));
    }
}

#[test]
fn predict_single_tree() {
    let (model, input, expected) = load_gbtree("gbtree_single_tree");
    let forest = model.to_forest().expect("conversion failed");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = forest.predict_row(features);
        assert_preds_match(&pred, &[expected_preds[i]], DEFAULT_TOLERANCE_F64, &format!("row {i}"));
    }
}

#[test]
fn predict_many_trees() {
    let (model, input, expected) = load_gbtree("gbtree_many_trees");
    let forest = model.to_forest().expect("conversion failed");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = forest.predict_row(features);
        assert_preds_match(&pred, &[expected_preds[i]], DEFAULT_TOLERANCE_F64, &format!("row {i}"));
    }
}

#[test]
fn predict_wide_features() {
    let (model, input, expected) = load_gbtree("gbtree_wide_features");
    let forest = model.to_forest().expect("conversion failed");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = forest.predict_row(features);
        assert_preds_match(&pred, &[expected_preds[i]], DEFAULT_TOLERANCE_F64, &format!("row {i}"));
    }
}

#[test]
fn predict_batch_regression() {
    let (model, input, expected) = load_gbtree("gbtree_regression");
    let forest = model.to_forest().expect("conversion failed");

    let rows = input.to_f32_rows();
    let expected_preds = expected.as_flat();

    let feature_refs: Vec<&[f32]> = rows.iter().map(|v| v.as_slice()).collect();
    let predictions = forest.predict_batch(&feature_refs);

    assert_eq!(predictions.len(), expected_preds.len());
    for (i, (pred, exp)) in predictions.iter().zip(expected_preds.iter()).enumerate() {
        assert_preds_match(pred, &[*exp], DEFAULT_TOLERANCE_F64, &format!("batch row {i}"));
    }
}

// =============================================================================
// Categorical Feature Tests
// =============================================================================

#[test]
fn parse_categorical_model() {
    let (model, _, _) = load_gbtree("gbtree_categorical");

    assert!(!model.learner.feature_types.is_empty());
    let has_categorical = model
        .learner
        .feature_types
        .iter()
        .any(|ft| matches!(ft, booste_rs::compat::xgboost::FeatureType::Categorical));
    assert!(has_categorical, "Expected categorical feature type");
}

#[test]
fn convert_categorical_model() {
    let (model, _, _) = load_gbtree("gbtree_categorical");
    let forest = model.to_forest().expect("conversion failed");

    assert_eq!(forest.num_groups(), 1);
    assert!(forest.num_trees() > 0);

    // Check that at least one tree has categorical splits
    let has_categorical = (0..forest.num_trees()).any(|i| forest.tree(i).has_categorical());
    assert!(has_categorical, "Expected at least one tree with categorical splits");
}

#[test]
fn predict_categorical() {
    let (model, input, expected) = load_gbtree("gbtree_categorical");
    let forest = model.to_forest().expect("conversion failed");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = forest.predict_row(features);
        assert_preds_match(&pred, &[expected_preds[i]], DEFAULT_TOLERANCE_F64, &format!("row {i}"));
    }
}

#[test]
fn parse_categorical_binary_model() {
    let (model, _, _) = load_gbtree("gbtree_categorical_binary");

    let has_categorical = model
        .learner
        .feature_types
        .iter()
        .any(|ft| matches!(ft, booste_rs::compat::xgboost::FeatureType::Categorical));
    assert!(has_categorical, "Expected categorical feature type");
}

#[test]
fn predict_categorical_binary() {
    let (model, input, expected) = load_gbtree("gbtree_categorical_binary");
    let forest = model.to_forest().expect("conversion failed");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = forest.predict_row(features);
        assert_preds_match(&pred, &[expected_preds[i]], DEFAULT_TOLERANCE_F64, &format!("row {i}"));
    }
}

// =============================================================================
// DART Tests
// =============================================================================

#[test]
fn parse_dart_model() {
    let (model, input, _) = load_dart("dart_regression");
    assert_eq!(input.num_features, 3);
    assert!(matches!(
        model.learner.gradient_booster,
        booste_rs::compat::xgboost::GradientBooster::Dart { .. }
    ));
}

#[test]
fn convert_dart_model() {
    let (model, _, _) = load_dart("dart_regression");
    let forest = model.to_forest().expect("conversion failed");
    assert_eq!(forest.num_groups(), 1);
}

#[test]
fn predict_dart() {
    let (model, input, expected) = load_dart("dart_regression");
    let forest = model.to_forest().expect("conversion failed");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = forest.predict_row(features);
        assert_preds_match(&pred, &[expected_preds[i]], DEFAULT_TOLERANCE_F64, &format!("row {i}"));
    }
}

// =============================================================================
// GBLinear Tests
// =============================================================================

#[test]
fn parse_gblinear_regression_model() {
    let (model, input, _) = load_gblinear("gblinear_regression");
    assert_eq!(input.num_features, 5);
    assert_eq!(input.num_rows, 10);
    assert!(model.is_linear());
}

#[test]
fn parse_gblinear_multiclass_model() {
    let (model, input, _) = load_gblinear("gblinear_multiclass");
    assert_eq!(input.num_features, 4);
    assert_eq!(model.learner.learner_model_param.num_class, 3);
    assert!(model.is_linear());
}

#[test]
fn predict_gblinear_regression() {
    use booste_rs::model::Booster;

    let (model, input, expected) = load_gblinear("gblinear_regression");
    let booster = model.to_booster().expect("conversion failed");
    let base_score = model.learner.learner_model_param.base_score;

    let linear = match booster {
        Booster::Linear(l) => l,
        _ => panic!("Expected linear booster"),
    };

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = linear.predict_row(features, &[base_score]);
        assert_preds_match(&pred, &[expected_preds[i]], DEFAULT_TOLERANCE_F64, &format!("row {i}"));
    }
}

#[test]
fn predict_gblinear_binary() {
    use booste_rs::model::Booster;

    let (model, input, expected) = load_gblinear("gblinear_binary");
    let booster = model.to_booster().expect("conversion failed");

    let prob_base_score = model.learner.learner_model_param.base_score;
    let objective = model.learner.objective.name();
    let margin_base_score = prob_to_margin(prob_base_score, objective);

    let linear = match booster {
        Booster::Linear(l) => l,
        _ => panic!("Expected linear booster"),
    };

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = linear.predict_row(features, &[margin_base_score]);
        assert_preds_match(&pred, &[expected_preds[i]], DEFAULT_TOLERANCE_F64, &format!("row {i}"));
    }
}

#[test]
fn predict_gblinear_multiclass() {
    use booste_rs::model::Booster;

    let (model, input, expected) = load_gblinear("gblinear_multiclass");
    let booster = model.to_booster().expect("conversion failed");

    let base_score_single = model.learner.learner_model_param.base_score;
    let num_class = model.learner.learner_model_param.num_class as usize;
    let base_scores = vec![base_score_single; num_class];

    let linear = match booster {
        Booster::Linear(l) => l,
        _ => panic!("Expected linear booster"),
    };

    let expected_preds = expected.as_nested();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = linear.predict_row(features, &base_scores);
        assert_eq!(pred.len(), num_class);
        assert_preds_match(&pred, &expected_preds[i], DEFAULT_TOLERANCE_F64, &format!("row {i}"));
    }
}
