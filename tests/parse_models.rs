use std::fs::File;
use std::path::Path;

// No external test framework required; iterate fixture paths manually
use serde_json::Value;

use xgboost_rs::loaders::xgboost::format::test_parse_model;
#[test]
fn parse_all_models() {
    let fixtures = [
        "tests/models/gbtree_regression.json",
        "tests/models/gblinear_binary.json",
        "tests/models/gbtree_multiclass_softmax.json",
        "tests/models/gbtree_multiclass_softprob.json",
        "tests/models/dart_regression.json",
        "tests/models/gblinear_multiclass.json",
        "tests/models/gbtree_binary_logistic.json",
        "tests/models/gbtree_multiclass_blobs.json",
    ];
    let mut parsed = 0usize;
    for path_str in fixtures.iter() {
        let path = Path::new(path_str);
        let file = File::open(&path).unwrap_or_else(|e| panic!("cannot open {}: {}", path.display(), e));
        let value: Value = serde_json::from_reader(file)
            .unwrap_or_else(|e| panic!("invalid json {}: {}", path.display(), e));
        let parsed_model = test_parse_model(&value);
        assert!(parsed_model.is_ok(), "Failed to parse {}: {:?}", path.display(), parsed_model.err());
        parsed += 1;
    }
    assert!(parsed > 0, "No JSON fixtures were found in tests/models");
}
