//! Prediction layout and shape integration tests.
//!
//! These tests verify that:
//! - Prediction outputs use the expected shape: `Array2 (n_outputs, n_rows)`
//! - Outputs are contiguous in memory (fast paths for transforms/metrics)
//! - Basic edge cases work (empty input, single sample)

use boosters::data::Dataset;
use boosters::model::{GBDTModel, ModelMeta, OutputTransform};
use boosters::repr::gbdt::{Forest, ScalarLeaf};
use boosters::scalar_tree;
use ndarray::array;

// =============================================================================
// Test Helpers
// =============================================================================

fn make_simple_forest() -> Forest<ScalarLeaf> {
    let tree = scalar_tree! {
        0 => num(0, 0.5, L) -> 1, 2,
        1 => leaf(1.0),
        2 => num(1, 0.3, R) -> 3, 4,
        3 => leaf(2.0),
        4 => leaf(3.0),
    };

    let mut forest = Forest::for_regression().with_base_score(vec![0.0]);
    forest.push_tree(tree, 0);
    forest
}

fn make_multiclass_forest() -> Forest<ScalarLeaf> {
    // 3-class classification with one tree per class
    let tree0 = scalar_tree! {
        0 => num(0, 0.5, L) -> 1, 2,
        1 => leaf(0.1),
        2 => leaf(0.9),
    };
    let tree1 = scalar_tree! {
        0 => num(0, 0.5, L) -> 1, 2,
        1 => leaf(0.2),
        2 => leaf(0.8),
    };
    let tree2 = scalar_tree! {
        0 => num(0, 0.5, L) -> 1, 2,
        1 => leaf(0.3),
        2 => leaf(0.7),
    };

    let mut forest = Forest::new(3).with_base_score(vec![0.0, 0.0, 0.0]);
    forest.push_tree(tree0, 0);
    forest.push_tree(tree1, 1);
    forest.push_tree(tree2, 2);
    forest
}

// =============================================================================
// Regression Tests
// =============================================================================

#[test]
fn predict_regression_shape_and_values() {
    let forest = make_simple_forest();
    let meta = ModelMeta::for_regression(2);
    let model = GBDTModel::from_parts(forest, meta, OutputTransform::Identity);

    // Feature-major: [n_features=2, n_samples=3]
    // sample 0: [0.3, 0.5] → goes left → 1.0
    // sample 1: [0.7, 0.5] → goes right-right → 3.0
    // sample 2: [0.6, 0.1] → goes right-left → 2.0
    let features_fm = array![
        [0.3f32, 0.7, 0.6], // feature 0 values
        [0.5, 0.5, 0.1],    // feature 1 values
    ];
    let dataset = Dataset::from_array(features_fm.view(), None, None);
    let preds = model.predict(&dataset, 1);

    // Verify shape: [n_groups=1, n_samples=3]
    assert_eq!(preds.nrows(), 1);
    assert_eq!(preds.ncols(), 3);

    // Verify expected values (preds[group, sample])
    assert!((preds[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((preds[[0, 1]] - 3.0).abs() < 1e-6);
    assert!((preds[[0, 2]] - 2.0).abs() < 1e-6);
}

#[test]
fn predict_raw_shape_and_values() {
    let forest = make_simple_forest();
    let meta = ModelMeta::for_regression(2);
    let model = GBDTModel::from_parts(forest, meta, OutputTransform::Identity);

    // Feature-major: [n_features=2, n_samples=2]
    let features_fm = array![
        [0.3f32, 0.7], // feature 0 values
        [0.5, 0.5],    // feature 1 values
    ];
    let dataset = Dataset::from_array(features_fm.view(), None, None);
    let preds = model.predict_raw(&dataset, 1);

    // Verify shape: [n_groups=1, n_samples=2]
    assert_eq!(preds.nrows(), 1);
    assert_eq!(preds.ncols(), 2);

    // Verify values are correct (raw predictions without transform)
    assert!((preds[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((preds[[0, 1]] - 3.0).abs() < 1e-6);
}

// =============================================================================
// Multiclass Tests
// =============================================================================

#[test]
fn predict_multiclass_shape() {
    let forest = make_multiclass_forest();
    let meta = ModelMeta::for_multiclass(2, 3); // 2 features, 3 classes
    let model = GBDTModel::from_parts(forest, meta, OutputTransform::Softmax);

    // Feature-major: [n_features=2, n_samples=2]
    let features_fm = array![
        [0.3f32, 0.7], // feature 0 values (sample 0: goes left, sample 1: goes right)
        [0.5, 0.5],    // feature 1 values
    ];
    let dataset = Dataset::from_array(features_fm.view(), None, None);
    let preds = model.predict(&dataset, 1);

    // Array2 shape: (n_groups=3, n_samples=2)
    assert_eq!(preds.nrows(), 3);
    assert_eq!(preds.ncols(), 2);

    // Whole prediction buffer should be contiguous.
    assert!(preds.as_slice().is_some());

    // Each row (group) should be contiguous (all samples for one group).
    for group in 0..3 {
        let row = preds.row(group);
        assert_eq!(row.len(), 2, "Row {} should have 2 elements", group);
        assert!(
            row.as_slice().is_some(),
            "Row {} should be contiguous",
            group
        );
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn predict_empty_input() {
    let forest = make_simple_forest();
    let meta = ModelMeta::for_regression(2);
    let model = GBDTModel::from_parts(forest, meta, OutputTransform::Identity);

    // Feature-major: [n_features=2, n_samples=0]
    let features_fm = ndarray::Array2::<f32>::zeros((2, 0));
    let dataset = Dataset::from_array(features_fm.view(), None, None);
    let preds = model.predict(&dataset, 1);

    // Should have shape (n_groups=1, n_samples=0)
    assert_eq!(preds.nrows(), 1);
    assert_eq!(preds.ncols(), 0);
}

#[test]
fn predict_single_sample() {
    let forest = make_simple_forest();
    let meta = ModelMeta::for_regression(2);
    let model = GBDTModel::from_parts(forest, meta, OutputTransform::Identity);

    // Feature-major: [n_features=2, n_samples=1]
    let features_fm = array![[0.3f32], [0.5]];
    let dataset = Dataset::from_array(features_fm.view(), None, None);
    let preds = model.predict(&dataset, 1);

    assert_eq!(preds.nrows(), 1);
    assert_eq!(preds.ncols(), 1);
    assert!((preds[[0, 0]] - 1.0).abs() < 1e-6);
}
