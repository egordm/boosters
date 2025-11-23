use std::fs::File;
use std::path::PathBuf;

use serde_json::Value;

use xgboost_rs::loaders::xgboost::format::test_parse_model;



#[test]
fn loaders_xgboost_parse() {
    // Arrange: locate files and parse JSON
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/test-cases/xgboost-models");

    let cases: Vec<_> = glob::glob(&root.join("*.json").to_string_lossy())
        .expect("Failed to read glob pattern")
        .map(|entry| entry.expect("Glob entry error"))
        .map(|path| {
            // assert!(result.is_ok(), "Failed to parse {}: {:?}", path.display(), result.err());
            let json: Value = serde_json::from_reader(File::open(&path).unwrap()).unwrap();
            (path, json)
        })
        .collect();

    assert!(!cases.is_empty(), "No test cases found in {}", root.display());

    // Act & Assert
    for (path, json) in cases.iter() {
        let result = test_parse_model(json);
        assert!(result.is_ok(), "Failed to parse {}: {:?}", path.display(), result.err());
    }
}
