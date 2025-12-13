//! XGBoost inference tests: model loading, conversion, and prediction parity.
//!
//! These tests compare booste-rs predictions against expected outputs generated
//! by Python XGBoost fixtures under `tests/test-cases/xgboost`.

#![cfg(feature = "xgboost-compat")]

use std::path::PathBuf;

use booste_rs::compat::xgboost::{Booster, FeatureType, GradientBooster};
use booste_rs::compat::XgbModel;

use super::test_data::{load_json, xgboost_test_cases_dir, TestExpected, TestInput, DEFAULT_TOLERANCE_F64};

// =============================================================================
// Test Case Loading
// =============================================================================

fn gbtree_dir() -> PathBuf {
    xgboost_test_cases_dir().join("gbtree/inference")
}

fn gblinear_dir() -> PathBuf {
    xgboost_test_cases_dir().join("gblinear/inference")
}

fn dart_dir() -> PathBuf {
    xgboost_test_cases_dir().join("dart/inference")
}

fn load_from_dir(dir: &std::path::Path, name: &str) -> (XgbModel, TestInput, TestExpected) {
    let model: XgbModel = load_json(&dir.join(format!("{name}.model.json")));
    let input: TestInput = load_json(&dir.join(format!("{name}.input.json")));
    let expected: TestExpected = load_json(&dir.join(format!("{name}.expected.json")));
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

    let mismatches: Vec<_> = actual
        .iter()
        .zip(expected.iter())
        .enumerate()
        .filter_map(|(i, (&a, &e))| {
            let diff = (a as f64 - e).abs();
            let rel_tol = e.abs() * 1e-6;
            if diff >= tolerance && diff >= rel_tol {
                Some((i, a, e, diff))
            } else {
                None
            }
        })
        .collect();

    if !mismatches.is_empty() {
        let mut msg = format!("{context}: predictions mismatch\n");
        for (i, a, e, diff) in mismatches.iter().take(10) {
            msg.push_str(&format!("  [{i}]: got {a}, expected {e}, diff {diff}\n"));
        }
        if mismatches.len() > 10 {
            msg.push_str(&format!("  ... and {} more mismatches\n", mismatches.len() - 10));
        }
        panic!("{msg}");
    }
}

// =============================================================================
// GBTree Prediction Tests
// =============================================================================

#[test]
fn predict_regression() {
    let (model, input, expected) = load_gbtree("gbtree_regression");
    let forest = model.to_forest().expect("conversion failed");
    forest.validate().expect("forest should be valid");

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
    forest.validate().expect("forest should be valid");

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
    forest.validate().expect("forest should be valid");

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
    forest.validate().expect("forest should be valid");

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
    forest.validate().expect("forest should be valid");

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
    forest.validate().expect("forest should be valid");

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
    forest.validate().expect("forest should be valid");

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
    forest.validate().expect("forest should be valid");

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
fn parse_categorical_model_has_feature_type() {
    let (model, _, _) = load_gbtree("gbtree_categorical");

    assert!(!model.learner.feature_types.is_empty());
    let has_categorical = model
        .learner
        .feature_types
        .iter()
        .any(|ft| matches!(ft, FeatureType::Categorical));
    assert!(has_categorical, "Expected categorical feature type");
}

#[test]
fn predict_categorical() {
    let (model, input, expected) = load_gbtree("gbtree_categorical");
    let forest = model.to_forest().expect("conversion failed");
    forest.validate().expect("forest should be valid");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = forest.predict_row(features);
        assert_preds_match(&pred, &[expected_preds[i]], DEFAULT_TOLERANCE_F64, &format!("row {i}"));
    }
}

#[test]
fn predict_categorical_binary() {
    let (model, input, expected) = load_gbtree("gbtree_categorical_binary");
    let forest = model.to_forest().expect("conversion failed");
    forest.validate().expect("forest should be valid");

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
fn predict_dart() {
    let (model, input, expected) = load_dart("dart_regression");

    assert!(matches!(
        model.learner.gradient_booster,
        GradientBooster::Dart { .. }
    ));

    let forest = model.to_forest().expect("conversion failed");
    forest.validate().expect("forest should be valid");

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
fn predict_gblinear_regression() {
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
