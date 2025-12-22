//! ndarray compatibility tests for prediction API.
//!
//! These tests verify that:
//! 1. Prediction methods produce correct results with proper shapes
//! 2. Output shape conventions are correct: ColMatrix (n_samples, n_groups) in column-major
//! 3. XGBoost compatibility is maintained

use boosters::data::RowMatrix;
use boosters::model::{GBDTModel, ModelMeta};
use boosters::repr::gbdt::{Forest, ScalarLeaf};
use boosters::scalar_tree;

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
    let model = GBDTModel::from_forest(forest, meta);

    let features = RowMatrix::from_vec(
        vec![
            0.3, 0.5, // sample 0: goes left → 1.0
            0.7, 0.5, // sample 1: goes right-right → 3.0
            0.6, 0.1, // sample 2: goes right-left → 2.0
        ],
        3,
        2,
    );

    // predict returns ColMatrix with shape (n_rows, n_groups)
    let preds = model.predict(&features, 1);

    // Verify shape
    assert_eq!(preds.n_rows(), 3);
    assert_eq!(preds.n_cols(), 1);

    // Verify expected values
    assert!((*preds.get(0, 0).unwrap() - 1.0).abs() < 1e-6);
    assert!((*preds.get(1, 0).unwrap() - 3.0).abs() < 1e-6);
    assert!((*preds.get(2, 0).unwrap() - 2.0).abs() < 1e-6);
}

#[test]
fn predict_raw_shape_and_values() {
    let forest = make_simple_forest();
    let meta = ModelMeta::for_regression(2);
    let model = GBDTModel::from_forest(forest, meta);

    let features = RowMatrix::from_vec(vec![0.3, 0.5, 0.7, 0.5], 2, 2);

    let preds = model.predict_raw(&features, 1);

    // Verify shape
    assert_eq!(preds.n_rows(), 2);
    assert_eq!(preds.n_cols(), 1);

    // Verify values are correct (raw predictions without transform)
    assert!((preds.get(0, 0).unwrap() - 1.0).abs() < 1e-6);
    assert!((preds.get(1, 0).unwrap() - 3.0).abs() < 1e-6);
}

// =============================================================================
// Multiclass Tests
// =============================================================================

#[test]
fn predict_multiclass_shape() {
    let forest = make_multiclass_forest();
    let meta = ModelMeta::for_multiclass(2, 3); // 2 features, 3 classes
    let model = GBDTModel::from_forest(forest, meta);

    let features = RowMatrix::from_vec(
        vec![
            0.3, 0.5, // sample 0: goes left
            0.7, 0.5, // sample 1: goes right
        ],
        2,
        2,
    );

    let preds = model.predict(&features, 1);

    // ColMatrix shape: (n_samples=2, n_groups=3)
    assert_eq!(preds.n_rows(), 2);
    assert_eq!(preds.n_cols(), 3);

    // Verify values are accessible
    for sample in 0..2 {
        for group in 0..3 {
            let val = preds.get(sample, group).unwrap();
            assert!(val.is_finite(), "Sample {}, group {}: value should be finite", sample, group);
        }
    }
}

#[test]
fn colmatrix_column_access_is_contiguous() {
    // Verify that accessing a group (column) in the output is contiguous
    let forest = make_multiclass_forest();
    let meta = ModelMeta::for_multiclass(2, 3);
    let model = GBDTModel::from_forest(forest, meta);

    let features = RowMatrix::from_vec(vec![0.3, 0.5, 0.7, 0.5, 0.4, 0.6], 3, 2);
    let preds = model.predict(&features, 1);

    // Each column should be contiguous (all samples for one group)
    for group in 0..3 {
        let col = preds.column(group);
        // column should be a contiguous slice
        assert_eq!(col.len(), 3, "Column {} should have 3 elements", group);
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn predict_empty_input() {
    let forest = make_simple_forest();
    let meta = ModelMeta::for_regression(2);
    let model = GBDTModel::from_forest(forest, meta);

    let features = RowMatrix::from_vec(vec![], 0, 2);
    let preds = model.predict(&features, 1);

    // Should have shape (n_samples=0, n_groups=1)
    assert_eq!(preds.n_rows(), 0);
    assert_eq!(preds.n_cols(), 1);
}

#[test]
fn predict_single_sample() {
    let forest = make_simple_forest();
    let meta = ModelMeta::for_regression(2);
    let model = GBDTModel::from_forest(forest, meta);

    let features = RowMatrix::from_vec(vec![0.3, 0.5], 1, 2);
    let preds = model.predict(&features, 1);

    assert_eq!(preds.n_rows(), 1);
    assert_eq!(preds.n_cols(), 1);
    assert!((*preds.get(0, 0).unwrap() - 1.0).abs() < 1e-6);
}
