//! Model-level integration tests for XGBoost model loading and prediction.
//!
//! These tests verify that `Model::predict()` produces correct transformed outputs
//! (e.g., sigmoid for binary classification, softmax for multiclass) matching
//! Python XGBoost predictions.
//!
//! Test case structure:
//! - {name}.model.json    - XGBoost model file
//! - {name}.input.json    - Test input features
//! - {name}.expected.json - Expected predictions (raw + transformed)

#![cfg(feature = "xgboost-compat")]

use std::fs::File;
use std::path::PathBuf;

use serde::Deserialize;

use booste_rs::compat::XgbModel;
use booste_rs::data::DenseMatrix;
use booste_rs::model::{FeatureInfo, Model, ModelMeta, ModelSource};
use booste_rs::objective::Objective;

// =============================================================================
// Test case structures
// =============================================================================

#[derive(Debug, Deserialize)]
struct TestInput {
    features: Vec<Vec<Option<f64>>>,
    num_rows: usize,
    num_features: usize,
}

impl TestInput {
    /// Convert to flat f32 vec for DenseMatrix, mapping None to NaN.
    fn to_dense_matrix(&self) -> DenseMatrix<f32, Box<[f32]>> {
        let flat: Vec<f32> = self
            .features
            .iter()
            .flat_map(|row| {
                row.iter()
                    .map(|&x| x.map(|v| v as f32).unwrap_or(f32::NAN))
            })
            .collect();
        DenseMatrix::from_vec(flat, self.num_rows, self.num_features)
    }
}

#[derive(Debug, Deserialize)]
struct TestExpected {
    predictions: serde_json::Value,
    predictions_transformed: serde_json::Value,
    objective: String,
    num_class: u32,
}

fn test_cases_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test-cases/xgboost")
}

fn load_test_case(name: &str) -> (XgbModel, TestInput, TestExpected) {
    let dir = test_cases_dir();

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

/// Convert XGBoost objective string to native Objective enum.
fn parse_objective(objective: &str, num_class: u32) -> Objective {
    match objective {
        "reg:squarederror" | "reg:linear" => Objective::SquaredError,
        "reg:absoluteerror" => Objective::AbsoluteError,
        "binary:logistic" => Objective::BinaryLogistic,
        "binary:logitraw" => Objective::BinaryLogitRaw,
        "multi:softmax" => Objective::MultiSoftmax { num_class },
        "multi:softprob" => Objective::MultiSoftprob { num_class },
        "reg:gamma" => Objective::Gamma,
        "count:poisson" => Objective::Poisson,
        _ => Objective::Custom,
    }
}

/// Build a Model from XgbModel for testing.
fn build_model(xgb: &XgbModel, objective: Objective) -> Model {
    let booster = xgb.to_booster().expect("conversion failed");
    let num_features = xgb.learner.learner_model_param.num_feature as u32;

    // For tree-based models, get num_groups from forest
    // For linear models, use num_class (or 1 for regression/binary)
    let num_groups = match &booster {
        booste_rs::model::Booster::Tree(f) | booste_rs::model::Booster::Dart { forest: f, .. } => {
            f.num_groups()
        }
        booste_rs::model::Booster::Linear(l) => l.num_groups() as u32,
    };

    Model::new(
        booster,
        ModelMeta {
            num_features,
            num_groups,
            base_score: vec![0.5; num_groups as usize], // XGBoost default
            source: ModelSource::XGBoostJson {
                version: xgb.version,
            },
        },
        FeatureInfo::numeric(num_features as usize),
        objective,
    )
}

// =============================================================================
// Helper functions
// =============================================================================

const TOLERANCE: f64 = 1e-5;

fn assert_predictions_match(actual: &[f32], expected: &[f64], tolerance: f64, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: prediction count mismatch: got {}, expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (*a as f64 - *e).abs();
        assert!(
            diff < tolerance,
            "{context}[{i}] mismatch: got {a}, expected {e}, diff {diff}"
        );
    }
}

// =============================================================================
// Model::predict_raw() tests - raw margin scores
// =============================================================================

#[test]
fn model_predict_raw_regression() {
    let (xgb_model, input, expected) = load_test_case("regression");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict_raw(&features);

    let expected_preds: Vec<f64> = serde_json::from_value(expected.predictions).unwrap();

    assert_eq!(output.shape(), (input.num_rows, 1));
    for (i, exp) in expected_preds.iter().enumerate() {
        let actual = output.row(i)[0] as f64;
        let diff = (actual - exp).abs();
        assert!(
            diff < TOLERANCE,
            "row {i}: got {actual}, expected {exp}, diff {diff}"
        );
    }
}

#[test]
fn model_predict_raw_binary() {
    let (xgb_model, input, expected) = load_test_case("binary_logistic");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict_raw(&features);

    let expected_preds: Vec<f64> = serde_json::from_value(expected.predictions).unwrap();

    assert_eq!(output.shape(), (input.num_rows, 1));
    for (i, exp) in expected_preds.iter().enumerate() {
        let actual = output.row(i)[0] as f64;
        let diff = (actual - exp).abs();
        assert!(
            diff < TOLERANCE,
            "row {i}: got {actual}, expected {exp}, diff {diff}"
        );
    }
}

#[test]
fn model_predict_raw_multiclass() {
    let (xgb_model, input, expected) = load_test_case("multiclass");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict_raw(&features);

    let expected_preds: Vec<Vec<f64>> = serde_json::from_value(expected.predictions).unwrap();

    assert_eq!(output.shape(), (input.num_rows, expected.num_class as usize));
    for (i, exp_row) in expected_preds.iter().enumerate() {
        assert_predictions_match(output.row(i), exp_row, TOLERANCE, &format!("row {i}"));
    }
}

// =============================================================================
// Model::predict() tests - with objective transforms
// =============================================================================

#[test]
fn model_predict_regression_transform() {
    let (xgb_model, input, expected) = load_test_case("regression");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    // For regression, transformed = raw (no transform)
    let expected_preds: Vec<f64> =
        serde_json::from_value(expected.predictions_transformed).unwrap();

    assert_eq!(output.shape(), (input.num_rows, 1));
    for (i, exp) in expected_preds.iter().enumerate() {
        let actual = output.row(i)[0] as f64;
        let diff = (actual - exp).abs();
        assert!(
            diff < TOLERANCE,
            "row {i}: got {actual}, expected {exp}, diff {diff}"
        );
    }
}

#[test]
fn model_predict_binary_sigmoid() {
    let (xgb_model, input, expected) = load_test_case("binary_logistic");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    // Binary logistic should apply sigmoid
    let expected_preds: Vec<f64> =
        serde_json::from_value(expected.predictions_transformed).unwrap();

    assert_eq!(output.shape(), (input.num_rows, 1));
    for (i, exp) in expected_preds.iter().enumerate() {
        let actual = output.row(i)[0] as f64;
        let diff = (actual - exp).abs();
        assert!(
            diff < TOLERANCE,
            "row {i}: got {actual}, expected {exp}, diff {diff}"
        );
    }

    // Verify values are probabilities (0-1)
    for i in 0..output.num_rows() {
        let p = output.row(i)[0];
        assert!(p >= 0.0 && p <= 1.0, "probability out of range: {p}");
    }
}

#[test]
fn model_predict_multiclass_softmax() {
    let (xgb_model, input, expected) = load_test_case("multiclass");
    // Use MultiSoftprob to get probability distribution
    let objective = Objective::MultiSoftprob {
        num_class: expected.num_class,
    };
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    // Multiclass softprob should apply softmax
    let expected_preds: Vec<Vec<f64>> =
        serde_json::from_value(expected.predictions_transformed).unwrap();

    assert_eq!(output.shape(), (input.num_rows, expected.num_class as usize));
    for (i, exp_row) in expected_preds.iter().enumerate() {
        assert_predictions_match(
            output.row(i),
            exp_row,
            TOLERANCE,
            &format!("row {i} softmax"),
        );

        // Verify probabilities sum to 1
        let sum: f32 = output.row(i).iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "row {i}: probabilities sum to {sum}, expected 1.0"
        );
    }
}

#[test]
fn model_predict_with_missing_values() {
    let (xgb_model, input, expected) = load_test_case("regression_missing");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    let expected_preds: Vec<f64> =
        serde_json::from_value(expected.predictions_transformed).unwrap();

    assert_eq!(output.shape(), (input.num_rows, 1));
    for (i, exp) in expected_preds.iter().enumerate() {
        let actual = output.row(i)[0] as f64;
        let diff = (actual - exp).abs();
        assert!(
            diff < TOLERANCE,
            "row {i}: got {actual}, expected {exp}, diff {diff}"
        );
    }
}

// =============================================================================
// Objective transform verification
// =============================================================================

#[test]
fn objective_produces_correct_flags() {
    assert!(Objective::BinaryLogistic.produces_probabilities());
    assert!(!Objective::SquaredError.produces_probabilities());
    assert!(
        Objective::MultiSoftprob { num_class: 3 }.produces_probabilities()
    );
    assert!(
        !Objective::MultiSoftmax { num_class: 3 }.produces_probabilities()
    );
    assert!(
        Objective::MultiSoftmax { num_class: 3 }.produces_class_indices()
    );
}

// =============================================================================
// Additional coverage tests
// =============================================================================

#[test]
fn model_predict_binary_with_missing() {
    let (xgb_model, input, expected) = load_test_case("binary_logistic_missing");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    let expected_preds: Vec<f64> =
        serde_json::from_value(expected.predictions_transformed).unwrap();

    assert_eq!(output.shape(), (input.num_rows, 1));
    for (i, exp) in expected_preds.iter().enumerate() {
        let actual = output.row(i)[0] as f64;
        let diff = (actual - exp).abs();
        assert!(
            diff < TOLERANCE,
            "row {i}: got {actual}, expected {exp}, diff {diff}"
        );
    }

    // Verify values are probabilities (0-1)
    for i in 0..output.num_rows() {
        let p = output.row(i)[0];
        assert!(p >= 0.0 && p <= 1.0, "probability out of range: {p}");
    }
}

#[test]
fn model_predict_multiclass_with_missing() {
    let (xgb_model, input, expected) = load_test_case("multiclass_missing");
    let objective = Objective::MultiSoftprob {
        num_class: expected.num_class,
    };
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    let expected_preds: Vec<Vec<f64>> =
        serde_json::from_value(expected.predictions_transformed).unwrap();

    assert_eq!(output.shape(), (input.num_rows, expected.num_class as usize));
    for (i, exp_row) in expected_preds.iter().enumerate() {
        assert_predictions_match(
            output.row(i),
            exp_row,
            TOLERANCE,
            &format!("row {i} multiclass_missing"),
        );

        // Verify probabilities sum to 1
        let sum: f32 = output.row(i).iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "row {i}: probabilities sum to {sum}, expected 1.0"
        );
    }
}

#[test]
fn model_predict_deep_trees() {
    let (xgb_model, input, expected) = load_test_case("deep_trees");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    let expected_preds: Vec<f64> =
        serde_json::from_value(expected.predictions_transformed).unwrap();

    assert_eq!(output.shape(), (input.num_rows, 1));
    for (i, exp) in expected_preds.iter().enumerate() {
        let actual = output.row(i)[0] as f64;
        let diff = (actual - exp).abs();
        assert!(
            diff < TOLERANCE,
            "deep_trees row {i}: got {actual}, expected {exp}, diff {diff}"
        );
    }
}

#[test]
fn model_predict_single_tree() {
    let (xgb_model, input, expected) = load_test_case("single_tree");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    let expected_preds: Vec<f64> =
        serde_json::from_value(expected.predictions_transformed).unwrap();

    assert_eq!(output.shape(), (input.num_rows, 1));
    assert_eq!(model.booster.num_trees(), 1);
    
    for (i, exp) in expected_preds.iter().enumerate() {
        let actual = output.row(i)[0] as f64;
        let diff = (actual - exp).abs();
        assert!(
            diff < TOLERANCE,
            "single_tree row {i}: got {actual}, expected {exp}, diff {diff}"
        );
    }
}

#[test]
fn model_predict_many_trees() {
    let (xgb_model, input, expected) = load_test_case("many_trees");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    let expected_preds: Vec<f64> =
        serde_json::from_value(expected.predictions_transformed).unwrap();

    assert_eq!(output.shape(), (input.num_rows, 1));
    assert_eq!(model.booster.num_trees(), 50);
    
    for (i, exp) in expected_preds.iter().enumerate() {
        let actual = output.row(i)[0] as f64;
        let diff = (actual - exp).abs();
        assert!(
            diff < TOLERANCE,
            "many_trees row {i}: got {actual}, expected {exp}, diff {diff}"
        );
    }
}

#[test]
fn model_predict_wide_features() {
    let (xgb_model, input, expected) = load_test_case("wide_features");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    assert_eq!(model.num_features(), 100);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    let expected_preds: Vec<f64> =
        serde_json::from_value(expected.predictions_transformed).unwrap();

    assert_eq!(output.shape(), (input.num_rows, 1));
    for (i, exp) in expected_preds.iter().enumerate() {
        let actual = output.row(i)[0] as f64;
        let diff = (actual - exp).abs();
        assert!(
            diff < TOLERANCE,
            "wide_features row {i}: got {actual}, expected {exp}, diff {diff}"
        );
    }
}

// =============================================================================
// DART booster tests
// =============================================================================

#[test]
fn model_predict_dart_regression() {
    let (xgb_model, input, expected) = load_test_case("dart_regression");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    // Verify it's actually a DART model
    assert!(xgb_model.is_dart());
    match &model.booster {
        booste_rs::model::Booster::Dart { weights, .. } => {
            assert_eq!(weights.len(), model.booster.num_trees());
        }
        _ => panic!("Expected DART booster"),
    }

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    let expected_preds: Vec<f64> =
        serde_json::from_value(expected.predictions_transformed).unwrap();

    assert_eq!(output.shape(), (input.num_rows, 1));
    for (i, exp) in expected_preds.iter().enumerate() {
        let actual = output.row(i)[0] as f64;
        let diff = (actual - exp).abs();
        assert!(
            diff < TOLERANCE,
            "dart_regression row {i}: got {actual}, expected {exp}, diff {diff}"
        );
    }
}

#[test]
fn model_predict_dart_raw_matches_expected() {
    let (xgb_model, input, expected) = load_test_case("dart_regression");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict_raw(&features);

    // For regression, raw and transformed should be the same
    let expected_preds: Vec<f64> = serde_json::from_value(expected.predictions).unwrap();

    assert_eq!(output.shape(), (input.num_rows, 1));
    for (i, exp) in expected_preds.iter().enumerate() {
        let actual = output.row(i)[0] as f64;
        let diff = (actual - exp).abs();
        assert!(
            diff < TOLERANCE,
            "dart_regression raw row {i}: got {actual}, expected {exp}, diff {diff}"
        );
    }
}
