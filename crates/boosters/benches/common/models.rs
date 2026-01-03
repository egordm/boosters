use std::path::PathBuf;

use boosters::model::{GBDTModel, GBLinearModel};
use boosters::persist::Model;
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

/// Path to benchmark models directory (native format).
pub fn bench_models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test-cases/persist/benchmark")
}

/// Path to original benchmark models directory (for LightGBM .lgb.txt files).
///
/// Used when benchmarking against native LightGBM library which requires its
/// original text format.
pub fn original_bench_models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test-cases/benchmark")
}

/// Load a booste-rs GBDT model from native format.
pub fn load_boosters_model(name: &str) -> LoadedForestModel {
    let path = bench_models_dir().join(format!("{name}.model.bstr.json"));
    let model = Model::load_json(&path).unwrap_or_else(|e| panic!("Failed to load {path:?}: {e}"));

    let gbdt = model
        .into_gbdt()
        .expect("Expected GBDT model for forest benchmark");
    let n_features = gbdt.meta().n_features;
    let forest = gbdt.forest().clone();

    LoadedForestModel { forest, n_features }
}

/// Load a booste-rs GBLinear model from native format.
pub fn load_linear_model(name: &str) -> LoadedLinearModel {
    let path = bench_models_dir().join(format!("{name}.model.bstr.json"));
    let model = Model::load_json(&path).unwrap_or_else(|e| panic!("Failed to load {path:?}: {e}"));

    let gblinear = model
        .into_gblinear()
        .expect("Expected GBLinear model for linear benchmark");
    let n_features = gblinear.meta().n_features;
    let linear = gblinear.linear().clone();

    LoadedLinearModel {
        model: linear,
        n_features,
    }
}
