//! Integration tests for the native persist format.
//!
//! These tests load models from `.model.bstr.json` fixtures and verify
//! prediction parity with expected outputs.

use std::fs::File;
use std::path::PathBuf;

use serde::de::DeserializeOwned;

use boosters::inference::gbdt::SimplePredictor;
use boosters::persist::Model;
use boosters::repr::gbdt::Forest;
use boosters::repr::gblinear::LinearModel;
use boosters::testing::{DEFAULT_TOLERANCE_F64, TestExpected, TestInput};

// =============================================================================
// Test Data Loading
// =============================================================================

fn test_cases_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test-cases/persist/v1")
}

fn gbtree_dir() -> PathBuf {
    test_cases_dir().join("gbtree/inference")
}

fn gblinear_dir() -> PathBuf {
    test_cases_dir().join("gblinear/inference")
}

fn dart_dir() -> PathBuf {
    test_cases_dir().join("dart/inference")
}

fn lightgbm_dir() -> PathBuf {
    test_cases_dir().join("lightgbm/inference")
}

fn load_json<T: DeserializeOwned>(path: &std::path::Path) -> T {
    let file =
        File::open(path).unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
    serde_json::from_reader(file)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()))
}

fn load_model_and_data(dir: &std::path::Path, name: &str) -> (Model, TestInput, TestExpected) {
    let json_path = dir.join(format!("{name}.model.bstr.json"));

    let model = Model::load_json(&json_path)
        .unwrap_or_else(|e| panic!("Failed to load model {name} from {json_path:?}: {e}"));
    let input: TestInput = load_json(&dir.join(format!("{name}.input.json")));
    let expected: TestExpected = load_json(&dir.join(format!("{name}.expected.json")));
    (model, input, expected)
}

fn load_gbtree(name: &str) -> (Forest, TestInput, TestExpected) {
    let (model, input, expected) = load_model_and_data(&gbtree_dir(), name);
    let forest = model
        .into_gbdt()
        .expect("expected GBDT model")
        .forest()
        .clone();
    (forest, input, expected)
}

fn load_gblinear(name: &str) -> (LinearModel, TestInput, TestExpected) {
    let (model, input, expected) = load_model_and_data(&gblinear_dir(), name);
    let linear = model
        .into_gblinear()
        .expect("expected GBLinear model")
        .linear()
        .clone();
    (linear, input, expected)
}

fn load_dart(name: &str) -> (Forest, TestInput, TestExpected) {
    let (model, input, expected) = load_model_and_data(&dart_dir(), name);
    let forest = model
        .into_gbdt()
        .expect("expected GBDT model")
        .forest()
        .clone();
    (forest, input, expected)
}

fn load_lightgbm(name: &str) -> (Forest, TestInput, TestExpected) {
    let (model, input, expected) = load_model_and_data(&lightgbm_dir().join(name), name);
    let forest = model
        .into_gbdt()
        .expect("expected GBDT model")
        .forest()
        .clone();
    (forest, input, expected)
}

// =============================================================================
// Helper Functions
// =============================================================================

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
            msg.push_str(&format!(
                "  ... and {} more mismatches\n",
                mismatches.len() - 10
            ));
        }
        panic!("{msg}");
    }
}

fn predict_row(forest: &Forest, features: &[f32]) -> Vec<f32> {
    let predictor = SimplePredictor::new(forest);
    let mut output = vec![0.0; predictor.n_groups()];
    let sample = ndarray::ArrayView1::from(features);
    predictor.predict_row_into(sample, None, &mut output);
    output
}

// =============================================================================
// GBDT (GBTree) Prediction Tests
// =============================================================================

#[test]
fn predict_regression() {
    let (forest, input, expected) = load_gbtree("gbtree_regression");
    forest.validate().expect("forest should be valid");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = predict_row(&forest, features);
        assert_preds_match(
            &pred,
            &[expected_preds[i]],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}

#[test]
fn predict_binary_logistic() {
    let (forest, input, expected) = load_gbtree("gbtree_binary_logistic");
    forest.validate().expect("forest should be valid");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = predict_row(&forest, features);
        assert_preds_match(
            &pred,
            &[expected_preds[i]],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}

#[test]
fn predict_multiclass() {
    let (forest, input, expected) = load_gbtree("gbtree_multiclass");
    forest.validate().expect("forest should be valid");

    let expected_preds = expected.as_nested();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = predict_row(&forest, features);
        assert_eq!(pred.len(), 3, "expected 3 classes");
        assert_preds_match(
            &pred,
            &expected_preds[i],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}

#[test]
fn predict_with_missing_values() {
    let (forest, input, expected) = load_gbtree("gbtree_regression_missing");
    forest.validate().expect("forest should be valid");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = predict_row(&forest, features);
        assert_preds_match(
            &pred,
            &[expected_preds[i]],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}

#[test]
fn predict_deep_trees() {
    let (forest, input, expected) = load_gbtree("gbtree_deep_trees");
    forest.validate().expect("forest should be valid");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = predict_row(&forest, features);
        assert_preds_match(
            &pred,
            &[expected_preds[i]],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}

#[test]
fn predict_single_tree() {
    let (forest, input, expected) = load_gbtree("gbtree_single_tree");
    forest.validate().expect("forest should be valid");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = predict_row(&forest, features);
        assert_preds_match(
            &pred,
            &[expected_preds[i]],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}

#[test]
fn predict_many_trees() {
    let (forest, input, expected) = load_gbtree("gbtree_many_trees");
    forest.validate().expect("forest should be valid");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = predict_row(&forest, features);
        assert_preds_match(
            &pred,
            &[expected_preds[i]],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}

#[test]
fn predict_wide_features() {
    let (forest, input, expected) = load_gbtree("gbtree_wide_features");
    forest.validate().expect("forest should be valid");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = predict_row(&forest, features);
        assert_preds_match(
            &pred,
            &[expected_preds[i]],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}

// =============================================================================
// Categorical Feature Tests
// =============================================================================

#[test]
fn predict_categorical() {
    let (forest, input, expected) = load_gbtree("gbtree_categorical");
    forest.validate().expect("forest should be valid");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = predict_row(&forest, features);
        assert_preds_match(
            &pred,
            &[expected_preds[i]],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}

#[test]
fn predict_categorical_binary() {
    let (forest, input, expected) = load_gbtree("gbtree_categorical_binary");
    forest.validate().expect("forest should be valid");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = predict_row(&forest, features);
        assert_preds_match(
            &pred,
            &[expected_preds[i]],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}

// =============================================================================
// DART Tests
// =============================================================================

#[test]
fn predict_dart() {
    let (forest, input, expected) = load_dart("dart_regression");
    forest.validate().expect("forest should be valid");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = predict_row(&forest, features);
        assert_preds_match(
            &pred,
            &[expected_preds[i]],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}

// =============================================================================
// GBLinear Tests
// =============================================================================

#[test]
fn predict_gblinear_regression() {
    let (linear, input, expected) = load_gblinear("gblinear_regression");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let mut pred = [0.0f32; 1];
        linear.predict_row_into(features, &mut pred);
        assert_preds_match(
            &pred,
            &[expected_preds[i]],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}

#[test]
fn predict_gblinear_binary() {
    let (linear, input, expected) = load_gblinear("gblinear_binary");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let mut pred = [0.0f32; 1];
        linear.predict_row_into(features, &mut pred);
        assert_preds_match(
            &pred,
            &[expected_preds[i]],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}

#[test]
fn predict_gblinear_multiclass() {
    let (linear, input, expected) = load_gblinear("gblinear_multiclass");

    let n_class = linear.n_groups();
    let expected_preds = expected.as_nested();

    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let mut pred = vec![0.0f32; n_class];
        linear.predict_row_into(features, &mut pred);
        assert_eq!(pred.len(), n_class);
        assert_preds_match(
            &pred,
            &expected_preds[i],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}

// =============================================================================
// LightGBM Fixtures (native format) Prediction Tests
// =============================================================================

#[test]
fn predict_lightgbm_small_tree() {
    let (forest, input, expected) = load_lightgbm("small_tree");
    forest.validate().expect("forest should be valid");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = predict_row(&forest, features);
        assert_preds_match(
            &pred,
            &[expected_preds[i]],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}

#[test]
fn predict_lightgbm_regression() {
    let (forest, input, expected) = load_lightgbm("regression");
    forest.validate().expect("forest should be valid");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = predict_row(&forest, features);
        assert_preds_match(
            &pred,
            &[expected_preds[i]],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}

#[test]
fn predict_lightgbm_regression_missing() {
    let (forest, input, expected) = load_lightgbm("regression_missing");
    forest.validate().expect("forest should be valid");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = predict_row(&forest, features);
        assert_preds_match(
            &pred,
            &[expected_preds[i]],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}

#[test]
fn predict_lightgbm_binary_classification() {
    let (forest, input, expected) = load_lightgbm("binary_classification");
    forest.validate().expect("forest should be valid");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = predict_row(&forest, features);
        assert_preds_match(
            &pred,
            &[expected_preds[i]],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}

#[test]
fn predict_lightgbm_multiclass() {
    let (forest, input, expected) = load_lightgbm("multiclass");
    forest.validate().expect("forest should be valid");

    let expected_preds = expected.as_nested();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = predict_row(&forest, features);
        assert_eq!(pred.len(), 3, "expected 3 classes");
        assert_preds_match(
            &pred,
            &expected_preds[i],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}

#[test]
fn predict_lightgbm_linear_tree() {
    let (forest, input, expected) = load_lightgbm("linear_tree");
    forest.validate().expect("forest should be valid");

    let expected_preds = expected.as_flat();
    for (i, features) in input.to_f32_rows().iter().enumerate() {
        let pred = predict_row(&forest, features);
        assert_preds_match(
            &pred,
            &[expected_preds[i]],
            DEFAULT_TOLERANCE_F64,
            &format!("row {i}"),
        );
    }
}
