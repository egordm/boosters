use std::fs::File;
use std::path::PathBuf;

use boosters::compat::xgboost::Booster;
use boosters::compat::XgbModel;
use boosters::repr::gbdt::{Forest, ScalarLeaf};
use boosters::repr::gblinear::LinearModel;

/// Minimal model wrapper for benchmarks.
///
/// Benchmarks typically need only the parsed tree ensemble plus metadata like feature count.
pub struct LoadedForestModel {
	pub forest: Forest<ScalarLeaf>,
	pub n_features: usize,
}

/// GBLinear model wrapper for benchmarks.
pub struct LoadedLinearModel {
	pub model: LinearModel,
	pub n_features: usize,
}

/// Path to benchmark models directory.
///
/// This matches the existing test-cases layout.
pub fn bench_models_dir() -> PathBuf {
	PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test-cases/benchmark")
}

/// Load a booste-rs GBDT model from JSON file.
pub fn load_boosters_model(name: &str) -> LoadedForestModel {
	let path = bench_models_dir().join(format!("{name}.model.json"));
	let file = File::open(&path).unwrap_or_else(|_| panic!("Failed to open model: {path:?}"));
	let xgb_model: XgbModel = serde_json::from_reader(file).expect("Failed to parse model");

	let forest = xgb_model.to_forest().expect("Failed to convert model to Forest");
	let n_features = xgb_model.learner.learner_model_param.n_features as usize;

	LoadedForestModel { forest, n_features }
}

/// Load a booste-rs GBLinear model from JSON file.
pub fn load_linear_model(name: &str) -> LoadedLinearModel {
	let path = bench_models_dir().join(format!("{name}.model.json"));
	let file = File::open(&path).unwrap_or_else(|_| panic!("Failed to open model: {path:?}"));
	let xgb_model: XgbModel = serde_json::from_reader(file).expect("Failed to parse model");

	let booster = xgb_model.to_booster().expect("Failed to convert model to Booster");
	let model = match booster {
		Booster::Linear(linear) => linear,
		_ => panic!("Expected GBLinear model but got tree-based model"),
	};
	let n_features = xgb_model.learner.learner_model_param.n_features as usize;

	LoadedLinearModel { model, n_features }
}
