use std::fs::File;
use std::path::PathBuf;

use booste_rs::compat::XgbModel;
use booste_rs::inference::gbdt::{Forest, ScalarLeaf};

/// Minimal model wrapper for benchmarks.
///
/// Benchmarks typically need only the parsed tree ensemble plus metadata like feature count.
pub struct LoadedForestModel {
	pub forest: Forest<ScalarLeaf>,
	pub num_features: usize,
}

/// Path to benchmark models directory.
///
/// This matches the existing test-cases layout.
pub fn bench_models_dir() -> PathBuf {
	PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test-cases/benchmark")
}

/// Load a booste-rs model from JSON file.
pub fn load_boosters_model(name: &str) -> LoadedForestModel {
	let path = bench_models_dir().join(format!("{name}.model.json"));
	let file = File::open(&path).unwrap_or_else(|_| panic!("Failed to open model: {path:?}"));
	let xgb_model: XgbModel = serde_json::from_reader(file).expect("Failed to parse model");

	let forest = xgb_model.to_forest().expect("Failed to convert model to Forest");
	let num_features = xgb_model.learner.learner_model_param.num_feature as usize;

	LoadedForestModel { forest, num_features }
}
