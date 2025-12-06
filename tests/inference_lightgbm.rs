//! Integration tests for LightGBM model compatibility.
//!
//! These tests verify that booste-rs can correctly load and predict
//! with LightGBM models across various configurations.

#![cfg(feature = "lightgbm-compat")]

use booste_rs::compat::lightgbm::LgbModel;

/// Test directory for LightGBM test cases.
const TEST_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/test-cases/lightgbm/inference"
);

mod regression {
    use super::*;

    #[test]
    fn load_and_predict_small_tree() {
        let model_path = format!("{}/small_tree/model.txt", TEST_DIR);
        let model = LgbModel::from_file(&model_path).expect("Failed to load model");
        let forest = model.to_forest().expect("Failed to convert model");

        assert_eq!(forest.num_groups(), 1);
        assert_eq!(forest.num_trees(), 3);

        // Test prediction on first row
        let features: Vec<f32> = vec![-0.234, 0.648, 1.523, -0.138, 0.497];
        let pred = forest.predict_row(&features);
        assert_eq!(pred.len(), 1);
        // Expected value from LightGBM: ~67.89
        assert!(pred[0] > 60.0 && pred[0] < 75.0);
    }

    #[test]
    fn load_and_predict_full_regression() {
        let model_path = format!("{}/regression/model.txt", TEST_DIR);
        let model = LgbModel::from_file(&model_path).expect("Failed to load model");
        let forest = model.to_forest().expect("Failed to convert model");

        assert_eq!(forest.num_groups(), 1);
        assert!(forest.num_trees() > 0);

        // Verify prediction produces reasonable output
        let features: Vec<f32> = vec![0.1; 10];
        let pred = forest.predict_row(&features);
        assert_eq!(pred.len(), 1);
        assert!(pred[0].is_finite());
    }
}

mod binary_classification {
    use super::*;

    #[test]
    fn load_and_predict_binary() {
        let model_path = format!("{}/binary_classification/model.txt", TEST_DIR);
        let model = LgbModel::from_file(&model_path).expect("Failed to load model");
        let forest = model.to_forest().expect("Failed to convert model");

        // Binary classification uses single output (raw logit)
        assert_eq!(forest.num_groups(), 1);
        assert!(forest.num_trees() > 0);

        // Verify raw prediction
        let features: Vec<f32> = vec![0.5; 20];
        let pred = forest.predict_row(&features);
        assert_eq!(pred.len(), 1);
        assert!(pred[0].is_finite());
    }
}

mod multiclass {
    use super::*;

    #[test]
    fn load_and_predict_multiclass() {
        let model_path = format!("{}/multiclass/model.txt", TEST_DIR);
        let model = LgbModel::from_file(&model_path).expect("Failed to load model");
        let forest = model.to_forest().expect("Failed to convert model");

        // 3-class classification uses 3 output groups
        assert_eq!(forest.num_groups(), 3);
        assert!(forest.num_trees() > 0);

        // Verify we get one output per class
        let features: Vec<f32> = vec![0.5; 20];
        let pred = forest.predict_row(&features);
        assert_eq!(pred.len(), 3);
        for p in &pred {
            assert!(p.is_finite());
        }
    }
}

mod missing_values {
    use super::*;

    #[test]
    fn load_and_predict_with_missing() {
        let model_path = format!("{}/regression_missing/model.txt", TEST_DIR);
        let model = LgbModel::from_file(&model_path).expect("Failed to load model");
        let forest = model.to_forest().expect("Failed to convert model");

        assert_eq!(forest.num_groups(), 1);

        // Test with NaN values (missing)
        let mut features: Vec<f32> = vec![0.5; 10];
        features[3] = f32::NAN;
        features[7] = f32::NAN;

        let pred = forest.predict_row(&features);
        assert_eq!(pred.len(), 1);
        // Prediction should be finite even with missing values
        assert!(pred[0].is_finite());
    }
}

mod model_metadata {
    use super::*;

    #[test]
    fn read_model_info() {
        let model_path = format!("{}/regression/model.txt", TEST_DIR);
        let model = LgbModel::from_file(&model_path).expect("Failed to load model");

        // Check model metadata
        assert_eq!(model.num_class(), 1);
        assert_eq!(model.header.num_tree_per_iteration, 1);
        assert!(!model.header.feature_names.is_empty());
        assert!(!model.trees.is_empty());
    }

    #[test]
    fn multiclass_model_info() {
        let model_path = format!("{}/multiclass/model.txt", TEST_DIR);
        let model = LgbModel::from_file(&model_path).expect("Failed to load model");

        assert_eq!(model.num_class(), 3);
        assert_eq!(model.header.num_tree_per_iteration, 3);
    }
}

mod batch_prediction {
    use super::*;

    #[test]
    fn predict_batch() {
        let model_path = format!("{}/regression/model.txt", TEST_DIR);
        let model = LgbModel::from_file(&model_path).expect("Failed to load model");
        let forest = model.to_forest().expect("Failed to convert model");

        // Create a small batch
        let batch: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![(i as f32) * 0.1; 10])
            .collect();

        // Predict each row
        let predictions: Vec<f32> = batch
            .iter()
            .map(|row| forest.predict_row(row)[0])
            .collect();

        assert_eq!(predictions.len(), 10);
        for p in &predictions {
            assert!(p.is_finite());
        }
    }
}
