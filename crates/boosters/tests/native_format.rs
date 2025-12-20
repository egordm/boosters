//! Integration tests for the native `.bstr` format.
//!
//! These tests verify format stability and cross-version compatibility.
//! Test fixtures are stored in `tests/test-cases/native/`.

#![cfg(feature = "storage")]

use boosters::repr::gbdt::{Forest, ScalarLeaf};
use boosters::repr::gblinear::LinearModel;
use boosters::scalar_tree;

const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/test-cases/native");

// ============================================================================
// Test Model Builders
// ============================================================================

/// Create a simple regression forest for testing.
fn simple_regression_forest() -> Forest<ScalarLeaf> {
    let tree = scalar_tree! {
        0 => num(0, 0.5, L) -> 1, 2,
        1 => leaf(1.0),
        2 => leaf(2.0),
    };

    let mut forest = Forest::for_regression().with_base_score(vec![0.5]);
    forest.push_tree(tree, 0);
    forest
}

/// Create a multi-tree forest for testing.
fn multi_tree_forest() -> Forest<ScalarLeaf> {
    let tree1 = scalar_tree! {
        0 => num(0, 0.5, L) -> 1, 2,
        1 => leaf(1.0),
        2 => leaf(2.0),
    };

    let tree2 = scalar_tree! {
        0 => num(1, 0.3, R) -> 1, 2,
        1 => num(0, 0.7, L) -> 3, 4,
        2 => leaf(3.0),
        3 => leaf(0.5),
        4 => leaf(1.5),
    };

    let mut forest = Forest::for_regression().with_base_score(vec![0.0]);
    forest.push_tree(tree1, 0);
    forest.push_tree(tree2, 0);
    forest
}

/// Create a forest with categorical splits.
fn categorical_forest() -> Forest<ScalarLeaf> {
    let tree = scalar_tree! {
        0 => cat(0, [1, 3, 5], L) -> 1, 2,
        1 => leaf(-1.0),
        2 => leaf(1.0),
    };

    let mut forest = Forest::for_regression();
    forest.push_tree(tree, 0);
    forest
}

/// Create a multi-class forest (n_groups > 1).
fn multiclass_forest() -> Forest<ScalarLeaf> {
    // 3-class classification: one tree per class
    let tree0 = scalar_tree! {
        0 => num(0, 0.5, L) -> 1, 2,
        1 => leaf(0.1),
        2 => leaf(-0.2),
    };

    let tree1 = scalar_tree! {
        0 => num(0, 0.5, L) -> 1, 2,
        1 => leaf(-0.1),
        2 => leaf(0.3),
    };

    let tree2 = scalar_tree! {
        0 => num(0, 0.5, L) -> 1, 2,
        1 => leaf(0.0),
        2 => leaf(-0.1),
    };

    let mut forest = Forest::new(3).with_base_score(vec![0.0, 0.0, 0.0]);
    forest.push_tree(tree0, 0);
    forest.push_tree(tree1, 1);
    forest.push_tree(tree2, 2);
    forest
}

/// Create a simple linear model for testing.
fn simple_linear_model() -> LinearModel {
    // 3 features, 1 group: weights [0.5, 0.3, -0.2] + bias [0.1]
    let weights = vec![0.5, 0.3, -0.2, 0.1].into_boxed_slice();
    LinearModel::new(weights, 3, 1)
}

/// Create a multi-output linear model.
fn multioutput_linear_model() -> LinearModel {
    // 2 features, 2 groups
    // Group 0: weights [0.5, 0.3] bias [0.1]
    // Group 1: weights [-0.2, 0.4] bias [-0.1]
    let weights = vec![
        0.5, 0.3, 0.1,   // group 0
        -0.2, 0.4, -0.1, // group 1
    ]
    .into_boxed_slice();
    LinearModel::new(weights, 2, 2)
}

// ============================================================================
// Roundtrip Tests (In-Memory)
// ============================================================================

#[test]
fn forest_simple_roundtrip() {
    let forest = simple_regression_forest();
    let bytes = forest.to_bytes().expect("serialize");
    let restored = Forest::from_bytes(&bytes).expect("deserialize");

    assert_eq!(forest.n_trees(), restored.n_trees());
    assert_eq!(forest.predict_row(&[0.3]), restored.predict_row(&[0.3]));
    assert_eq!(forest.predict_row(&[0.7]), restored.predict_row(&[0.7]));
}

#[test]
fn forest_multi_tree_roundtrip() {
    let forest = multi_tree_forest();
    let bytes = forest.to_bytes().expect("serialize");
    let restored = Forest::from_bytes(&bytes).expect("deserialize");

    assert_eq!(forest.n_trees(), restored.n_trees());
    for x in [0.2, 0.5, 0.8] {
        for y in [0.1, 0.4, 0.9] {
            let row = [x, y];
            assert_eq!(
                forest.predict_row(&row),
                restored.predict_row(&row),
                "prediction mismatch at {:?}",
                row
            );
        }
    }
}

#[test]
fn forest_categorical_roundtrip() {
    let forest = categorical_forest();
    let bytes = forest.to_bytes().expect("serialize");
    let restored = Forest::from_bytes(&bytes).expect("deserialize");

    // Categories 1,3,5 go right (value 1.0), others go left (value -1.0)
    for cat in 0..8 {
        let pred_orig = forest.predict_row(&[cat as f32]);
        let pred_rest = restored.predict_row(&[cat as f32]);
        assert_eq!(
            pred_orig, pred_rest,
            "prediction mismatch for category {}",
            cat
        );
    }
}

#[test]
fn forest_multiclass_roundtrip() {
    let forest = multiclass_forest();
    let bytes = forest.to_bytes().expect("serialize");
    let restored = Forest::from_bytes(&bytes).expect("deserialize");

    assert_eq!(forest.n_groups(), restored.n_groups());
    assert_eq!(forest.n_trees(), restored.n_trees());

    // Verify predictions for all 3 classes
    for x in [0.2, 0.7] {
        let row = [x];
        let pred_orig = forest.predict_row(&row);
        let pred_rest = restored.predict_row(&row);
        assert_eq!(pred_orig, pred_rest, "prediction mismatch at {:?}", row);
    }
}

#[test]
fn linear_simple_roundtrip() {
    let model = simple_linear_model();
    let bytes = model.to_bytes(100).expect("serialize");
    let restored = LinearModel::from_bytes(&bytes).expect("deserialize");

    assert_eq!(model.n_features(), restored.n_features());
    assert_eq!(model.n_groups(), restored.n_groups());

    // Check weights
    for f in 0..model.n_features() {
        assert_eq!(
            model.weight(f, 0),
            restored.weight(f, 0),
            "weight mismatch for feature {}",
            f
        );
    }
    assert_eq!(model.bias(0), restored.bias(0));
}

#[test]
fn linear_multioutput_roundtrip() {
    let model = multioutput_linear_model();
    let bytes = model.to_bytes(50).expect("serialize");
    let restored = LinearModel::from_bytes(&bytes).expect("deserialize");

    assert_eq!(model.n_groups(), restored.n_groups());
    for g in 0..model.n_groups() {
        for f in 0..model.n_features() {
            assert_eq!(
                model.weight(f, g),
                restored.weight(f, g),
                "weight mismatch at feature {}, group {}",
                f,
                g
            );
        }
        assert_eq!(model.bias(g), restored.bias(g), "bias mismatch at group {}", g);
    }
}

// ============================================================================
// File-Based Tests
// ============================================================================

#[test]
fn forest_file_roundtrip() {
    let forest = simple_regression_forest();
    let path = std::env::temp_dir().join("boosters_test_forest_integration.bstr");

    forest.save(&path).expect("save");
    let restored = Forest::load(&path).expect("load");

    std::fs::remove_file(&path).ok();

    assert_eq!(forest.predict_row(&[0.3]), restored.predict_row(&[0.3]));
}

#[test]
fn linear_file_roundtrip() {
    let model = simple_linear_model();
    let path = std::env::temp_dir().join("boosters_test_linear_integration.bstr");

    model.save(&path, 100).expect("save");
    let restored = LinearModel::load(&path).expect("load");

    std::fs::remove_file(&path).ok();

    assert_eq!(model.weight(0, 0), restored.weight(0, 0));
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn type_mismatch_forest_as_linear() {
    let forest = simple_regression_forest();
    let bytes = forest.to_bytes().expect("serialize");

    let result = LinearModel::from_bytes(&bytes);
    assert!(result.is_err(), "should reject forest bytes as LinearModel");
}

#[test]
fn type_mismatch_linear_as_forest() {
    let model = simple_linear_model();
    let bytes = model.to_bytes(100).expect("serialize");

    let result = Forest::from_bytes(&bytes);
    assert!(result.is_err(), "should reject linear bytes as Forest");
}

#[test]
fn corrupt_data_rejected() {
    let forest = simple_regression_forest();
    let mut bytes = forest.to_bytes().expect("serialize");

    // Corrupt a byte in the middle
    if bytes.len() > 50 {
        bytes[50] ^= 0xFF;
    }

    let result = Forest::from_bytes(&bytes);
    assert!(result.is_err(), "should reject corrupted data");
}

#[test]
fn empty_data_rejected() {
    let result = Forest::from_bytes(&[]);
    assert!(result.is_err(), "should reject empty data");
}

#[test]
fn truncated_data_rejected() {
    let forest = simple_regression_forest();
    let bytes = forest.to_bytes().expect("serialize");

    // Take only first half
    let truncated = &bytes[..bytes.len() / 2];

    let result = Forest::from_bytes(truncated);
    assert!(result.is_err(), "should reject truncated data");
}

// ============================================================================
// Format Stability Tests (Fixture-Based)
// ============================================================================

/// Generate fixture files for format stability testing.
///
/// Run with: `cargo test --features storage generate_fixtures -- --ignored`
#[test]
#[ignore]
fn generate_fixtures() {
    use std::fs;
    use std::path::Path;

    let fixture_dir = Path::new(FIXTURE_DIR);
    fs::create_dir_all(fixture_dir).expect("create fixture dir");

    // Generate fixtures
    let fixtures: Vec<(&str, Vec<u8>)> = vec![
        ("simple_forest.bstr", simple_regression_forest().to_bytes().unwrap()),
        ("multi_tree_forest.bstr", multi_tree_forest().to_bytes().unwrap()),
        ("categorical_forest.bstr", categorical_forest().to_bytes().unwrap()),
        ("multiclass_forest.bstr", multiclass_forest().to_bytes().unwrap()),
        ("simple_linear.bstr", simple_linear_model().to_bytes(100).unwrap()),
        ("multioutput_linear.bstr", multioutput_linear_model().to_bytes(50).unwrap()),
    ];

    for (name, bytes) in fixtures {
        let path = fixture_dir.join(name);
        fs::write(&path, bytes).expect("write fixture");
        println!("Generated: {}", path.display());
    }
}

/// Verify fixtures can still be loaded.
///
/// This test validates format backward compatibility.
#[test]
fn load_fixtures() {
    use std::path::Path;

    let fixture_dir = Path::new(FIXTURE_DIR);
    if !fixture_dir.exists() {
        eprintln!("Fixture directory not found, skipping fixture tests");
        eprintln!("Run `cargo test --features storage generate_fixtures -- --ignored` to generate");
        return;
    }

    // Test forest fixtures
    let forest_fixtures = ["simple_forest.bstr", "multi_tree_forest.bstr", "categorical_forest.bstr", "multiclass_forest.bstr"];
    for name in forest_fixtures {
        let path = fixture_dir.join(name);
        if path.exists() {
            let result = Forest::load(&path);
            assert!(result.is_ok(), "Failed to load fixture {}: {:?}", name, result.err());
        }
    }

    // Test linear fixtures
    let linear_fixtures = ["simple_linear.bstr", "multioutput_linear.bstr"];
    for name in linear_fixtures {
        let path = fixture_dir.join(name);
        if path.exists() {
            let result = LinearModel::load(&path);
            assert!(result.is_ok(), "Failed to load fixture {}: {:?}", name, result.err());
        }
    }
}
