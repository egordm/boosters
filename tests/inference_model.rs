//! Model API inference tests: high-level prediction with transforms.
//!
//! These tests verify that the `Model` API produces correct transformed outputs:
//! - `predict_raw()` - raw margin scores
//! - `predict()` - transformed outputs (sigmoid, softmax, etc.)
//!
//! Test cases organized by:
//! - Regression, binary classification, multiclass
//! - With/without missing values
//! - GBTree and DART boosters

#![cfg(feature = "xgboost-compat")]

use std::fs::File;
use std::path::PathBuf;

use approx::assert_abs_diff_eq;
use serde::Deserialize;

use booste_rs::compat::XgbModel;
use booste_rs::data::RowMatrix;
use booste_rs::model::{FeatureInfo, Model, ModelMeta, ModelSource};
use booste_rs::objective::Objective;
use booste_rs::predict::PredictionOutput;

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
    /// Convert to flat f32 vec for RowMatrix, mapping None to NaN.
    fn to_dense_matrix(&self) -> RowMatrix<f32, Box<[f32]>> {
        let flat: Vec<f32> = self
            .features
            .iter()
            .flat_map(|row| {
                row.iter()
                    .map(|&x| x.map(|v| v as f32).unwrap_or(f32::NAN))
            })
            .collect();
        RowMatrix::from_vec(flat, self.num_rows, self.num_features)
    }
}

#[derive(Debug, Deserialize)]
struct TestExpected {
    predictions: serde_json::Value,
    predictions_transformed: serde_json::Value,
    objective: String,
    num_class: u32,
}

// =============================================================================
// Test Case Loading
// =============================================================================

fn gbtree_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test-cases/xgboost/gbtree/inference")
}

fn dart_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test-cases/xgboost/dart/inference")
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

fn load_dart(name: &str) -> (XgbModel, TestInput, TestExpected) {
    load_from_dir(&dart_dir(), name)
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

/// Create expected PredictionOutput from scalar JSON array (single-group output).
fn expected_scalar_output(json: &serde_json::Value) -> PredictionOutput {
    let preds: Vec<f64> = serde_json::from_value(json.clone()).unwrap();
    let n = preds.len();
    PredictionOutput::new(preds.into_iter().map(|x| x as f32).collect(), n, 1)
}

/// Create expected PredictionOutput from nested JSON array (multi-group output).
fn expected_multigroup_output(json: &serde_json::Value, num_groups: usize) -> PredictionOutput {
    let preds: Vec<Vec<f64>> = serde_json::from_value(json.clone()).unwrap();
    let n = preds.len();
    let flat: Vec<f32> = preds.into_iter().flatten().map(|x| x as f32).collect();
    PredictionOutput::new(flat, n, num_groups)
}

// =============================================================================
// Model::predict_raw() tests - raw margin scores
// =============================================================================

#[test]
fn model_predict_raw_regression() {
    let (xgb_model, input, expected) = load_gbtree("gbtree_regression");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict_raw(&features);

    let expected_preds: Vec<f64> = serde_json::from_value(expected.predictions).unwrap();
    let expected_output = PredictionOutput::new(
        expected_preds.iter().map(|&x| x as f32).collect(),
        expected_preds.len(),
        1,
    );

    assert_abs_diff_eq!(output, expected_output, epsilon = TOLERANCE as f32);
}

#[test]
fn model_predict_raw_binary() {
    let (xgb_model, input, expected) = load_gbtree("gbtree_binary_logistic");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict_raw(&features);

    let expected_preds: Vec<f64> = serde_json::from_value(expected.predictions).unwrap();
    let expected_output = PredictionOutput::new(
        expected_preds.iter().map(|&x| x as f32).collect(),
        expected_preds.len(),
        1,
    );

    assert_abs_diff_eq!(output, expected_output, epsilon = TOLERANCE as f32);
}

#[test]
fn model_predict_raw_multiclass() {
    let (xgb_model, input, expected) = load_gbtree("gbtree_multiclass");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict_raw(&features);

    let expected_output =
        expected_multigroup_output(&expected.predictions, expected.num_class as usize);

    assert_abs_diff_eq!(output, expected_output, epsilon = TOLERANCE as f32);
}

// =============================================================================
// Model::predict() tests - with objective transforms
// =============================================================================

#[test]
fn model_predict_regression_transform() {
    let (xgb_model, input, expected) = load_gbtree("gbtree_regression");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    // For regression, transformed = raw (no transform)
    let expected_output = expected_scalar_output(&expected.predictions_transformed);

    assert_abs_diff_eq!(output, expected_output, epsilon = TOLERANCE as f32);
}

#[test]
fn model_predict_binary_sigmoid() {
    let (xgb_model, input, expected) = load_gbtree("gbtree_binary_logistic");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    // Binary logistic should apply sigmoid
    let expected_output = expected_scalar_output(&expected.predictions_transformed);

    assert_abs_diff_eq!(output, expected_output, epsilon = TOLERANCE as f32);

    // Verify values are probabilities (0-1)
    for i in 0..output.num_rows() {
        let p = output.row(i)[0];
        assert!((0.0..=1.0).contains(&p), "probability out of range: {p}");
    }
}

#[test]
fn model_predict_multiclass_softmax() {
    let (xgb_model, input, expected) = load_gbtree("gbtree_multiclass");
    // Use MultiSoftprob to get probability distribution
    let objective = Objective::MultiSoftprob {
        num_class: expected.num_class,
    };
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    // Multiclass softprob should apply softmax
    let expected_output =
        expected_multigroup_output(&expected.predictions_transformed, expected.num_class as usize);

    assert_abs_diff_eq!(output, expected_output, epsilon = TOLERANCE as f32);

    // Verify probabilities sum to 1 for each row
    for i in 0..output.num_rows() {
        let sum: f32 = output.row(i).iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "row {i}: probabilities sum to {sum}, expected 1.0"
        );
    }
}

#[test]
fn model_predict_with_missing_values() {
    let (xgb_model, input, expected) = load_gbtree("gbtree_regression_missing");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    let expected_output = expected_scalar_output(&expected.predictions_transformed);

    assert_abs_diff_eq!(output, expected_output, epsilon = TOLERANCE as f32);
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
    let (xgb_model, input, expected) = load_gbtree("gbtree_binary_logistic_missing");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    let expected_output = expected_scalar_output(&expected.predictions_transformed);

    assert_abs_diff_eq!(output, expected_output, epsilon = TOLERANCE as f32);

    // Verify values are probabilities (0-1)
    for i in 0..output.num_rows() {
        let p = output.row(i)[0];
        assert!((0.0..=1.0).contains(&p), "probability out of range: {p}");
    }
}

#[test]
fn model_predict_multiclass_with_missing() {
    let (xgb_model, input, expected) = load_gbtree("gbtree_multiclass_missing");
    let objective = Objective::MultiSoftprob {
        num_class: expected.num_class,
    };
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    let expected_output =
        expected_multigroup_output(&expected.predictions_transformed, expected.num_class as usize);

    assert_abs_diff_eq!(output, expected_output, epsilon = TOLERANCE as f32);

    // Verify probabilities sum to 1 for each row
    for i in 0..output.num_rows() {
        let sum: f32 = output.row(i).iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "row {i}: probabilities sum to {sum}, expected 1.0"
        );
    }
}

#[test]
fn model_predict_deep_trees() {
    let (xgb_model, input, expected) = load_gbtree("gbtree_deep_trees");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    let expected_output = expected_scalar_output(&expected.predictions_transformed);

    assert_abs_diff_eq!(output, expected_output, epsilon = TOLERANCE as f32);
}

#[test]
fn model_predict_single_tree() {
    let (xgb_model, input, expected) = load_gbtree("gbtree_single_tree");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    let expected_output = expected_scalar_output(&expected.predictions_transformed);

    assert_eq!(model.booster.num_trees(), 1);
    assert_abs_diff_eq!(output, expected_output, epsilon = TOLERANCE as f32);
}

#[test]
fn model_predict_many_trees() {
    let (xgb_model, input, expected) = load_gbtree("gbtree_many_trees");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    let expected_output = expected_scalar_output(&expected.predictions_transformed);

    assert_eq!(model.booster.num_trees(), 50);
    assert_abs_diff_eq!(output, expected_output, epsilon = TOLERANCE as f32);
}

#[test]
fn model_predict_wide_features() {
    let (xgb_model, input, expected) = load_gbtree("gbtree_wide_features");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    assert_eq!(model.num_features(), 100);

    let features = input.to_dense_matrix();
    let output = model.predict(&features);

    let expected_output = expected_scalar_output(&expected.predictions_transformed);

    assert_abs_diff_eq!(output, expected_output, epsilon = TOLERANCE as f32);
}

// =============================================================================
// DART booster tests
// =============================================================================

#[test]
fn model_predict_dart_regression() {
    let (xgb_model, input, expected) = load_dart("dart_regression");
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

    let expected_output = expected_scalar_output(&expected.predictions_transformed);

    assert_abs_diff_eq!(output, expected_output, epsilon = TOLERANCE as f32);
}

#[test]
fn model_predict_dart_raw_matches_expected() {
    let (xgb_model, input, expected) = load_dart("dart_regression");
    let objective = parse_objective(&expected.objective, expected.num_class);
    let model = build_model(&xgb_model, objective);

    let features = input.to_dense_matrix();
    let output = model.predict_raw(&features);

    // For regression, raw and transformed should be the same
    let expected_output = expected_scalar_output(&expected.predictions);

    assert_abs_diff_eq!(output, expected_output, epsilon = TOLERANCE as f32);
}
