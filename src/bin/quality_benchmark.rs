//! Comprehensive quality benchmark runner.
//!
//! This script runs quality evaluations across multiple configurations and seeds,
//! aggregating results with confidence intervals.
//!
//! Usage:
//!   cargo run --bin quality_benchmark --release --features "bench-xgboost,bench-lightgbm,io-parquet" -- [options]
//!
//! Options:
//!   --seeds <n>          Number of seeds to run (default: 5)
//!   --out <path>         Output markdown file (default: stdout)
//!   --quick              Quick mode: fewer rows, fewer trees
//!   --mode <mode>        Benchmark mode: all (default), synthetic, real
//!   --no-real            Alias for --mode synthetic (deprecated)
//!   --libsvm <path>      Add libsvm regression dataset (label + index:value, 1-based)
//!   --uci-machine <path> Add UCI machine.data regression dataset
//!   --label0 <path>      Add label0 dataset (tab/space-separated: label first)
//!
//! Real-world Datasets:
//!   By default, if parquet files exist in data/benchmarks/, they are included:
//!   - california_housing.parquet (regression, 20k samples)
//!   - adult.parquet (binary classification, 48k samples)
//!   - covertype.parquet (multiclass, 581k samples - subsampled to 50k)
//!
//!   Generate these files with:
//!     cd tools/data_generation && uv run python scripts/generate_benchmark_datasets.py
//!
//! Examples:
//!   # Full benchmark (synthetic + real-world)
//!   cargo run --bin quality_benchmark --release --features "bench-xgboost,bench-lightgbm,io-parquet" -- \
//!       --seeds 5 --out docs/benchmarks/quality-report.md
//!
//!   # Synthetic only
//!   cargo run --bin quality_benchmark --release --features "bench-xgboost,bench-lightgbm" -- \
//!       --mode synthetic --seeds 3
//!
//!   # Real-world only  
//!   cargo run --bin quality_benchmark --release --features "bench-xgboost,bench-lightgbm,io-parquet" -- \
//!       --mode real --seeds 5

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use booste_rs::data::binned::BinnedDatasetBuilder;
use booste_rs::data::{ColMatrix, DenseMatrix, RowMajor, RowMatrix};
use booste_rs::inference::gbdt::{Predictor, UnrolledTraversal6};
use booste_rs::testing::data::{
	random_dense_f32, split_indices, synthetic_binary_targets_from_linear_score,
	synthetic_multiclass_targets_from_linear_scores, synthetic_regression_targets_linear,
};
use booste_rs::training::{
	Accuracy, GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, LinearLeafConfig, LogLoss, LogisticLoss, Mae,
	Metric, MulticlassAccuracy, MulticlassLogLoss, Objective, Rmse, SoftmaxLoss, SquaredLoss,
};

#[cfg(feature = "io-parquet")]
use booste_rs::data::io::parquet::load_parquet_xy_row_major_f32;

#[cfg(feature = "bench-xgboost")]
use xgb::parameters::tree::{GrowPolicy, TreeBoosterParametersBuilder, TreeMethod};
#[cfg(feature = "bench-xgboost")]
use xgb::parameters::{
	learning::LearningTaskParametersBuilder, learning::Objective as XgbObjective, BoosterParametersBuilder,
	BoosterType, TrainingParametersBuilder,
};
#[cfg(feature = "bench-xgboost")]
use xgb::{Booster as XgbBooster, DMatrix};

use serde::{Deserialize, Serialize};

// =============================================================================
// Task Types
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Task {
	Regression,
	Binary,
	Multiclass,
}

impl Task {
	fn as_str(&self) -> &'static str {
		match self {
			Task::Regression => "regression",
			Task::Binary => "binary",
			Task::Multiclass => "multiclass",
		}
	}
}

// =============================================================================
// Configuration
// =============================================================================

#[derive(Debug, Clone)]
enum DataSource {
	Synthetic { rows: usize, cols: usize },
	#[cfg(feature = "io-parquet")]
	Parquet { path: PathBuf, subsample: Option<usize> },
	LibSvm(PathBuf),
	UciMachine(PathBuf),
	Label0(PathBuf),
}

#[derive(Debug, Clone)]
struct BenchmarkConfig {
	name: String,
	task: Task,
	data_source: DataSource,
	trees: u32,
	depth: u32,
	classes: Option<usize>,
	/// Enable linear GBDT training (booste-rs and LightGBM only).
	linear_leaves: bool,
}

fn default_configs(quick: bool) -> Vec<BenchmarkConfig> {
	let (small_rows, medium_rows) = if quick { (2_000, 10_000) } else { (10_000, 50_000) };
	let trees = if quick { 50 } else { 100 };

	vec![
		// Regression benchmarks
		BenchmarkConfig {
			name: "regression_small".to_string(),
			task: Task::Regression,
			data_source: DataSource::Synthetic { rows: small_rows, cols: 50 },
			trees,
			depth: 6,
			classes: None,
			linear_leaves: false,
		},
		BenchmarkConfig {
			name: "regression_medium".to_string(),
			task: Task::Regression,
			data_source: DataSource::Synthetic { rows: medium_rows, cols: 100 },
			trees,
			depth: 6,
			classes: None,
			linear_leaves: false,
		},
		// Linear GBDT regression benchmarks (booste-rs + LightGBM only)
		BenchmarkConfig {
			name: "regression_linear_small".to_string(),
			task: Task::Regression,
			data_source: DataSource::Synthetic { rows: small_rows, cols: 50 },
			trees,
			depth: 6,
			classes: None,
			linear_leaves: true,
		},
		BenchmarkConfig {
			name: "regression_linear_medium".to_string(),
			task: Task::Regression,
			data_source: DataSource::Synthetic { rows: medium_rows, cols: 100 },
			trees,
			depth: 6,
			classes: None,
			linear_leaves: true,
		},
		// Binary classification benchmarks
		BenchmarkConfig {
			name: "binary_small".to_string(),
			task: Task::Binary,
			data_source: DataSource::Synthetic { rows: small_rows, cols: 50 },
			trees,
			depth: 6,
			classes: None,
			linear_leaves: false,
		},
		BenchmarkConfig {
			name: "binary_medium".to_string(),
			task: Task::Binary,
			data_source: DataSource::Synthetic { rows: medium_rows, cols: 100 },
			trees,
			depth: 6,
			classes: None,
			linear_leaves: false,
		},
		// Multiclass classification benchmarks
		BenchmarkConfig {
			name: "multiclass_small".to_string(),
			task: Task::Multiclass,
			data_source: DataSource::Synthetic { rows: small_rows, cols: 50 },
			trees,
			depth: 6,
			classes: Some(5),
			linear_leaves: false,
		},
		BenchmarkConfig {
			name: "multiclass_medium".to_string(),
			task: Task::Multiclass,
			data_source: DataSource::Synthetic { rows: medium_rows, cols: 100 },
			trees,
			depth: 6,
			classes: Some(5),
			linear_leaves: false,
		},
	]
}

/// Default paths for real-world benchmark datasets.
#[cfg(feature = "io-parquet")]
const CALIFORNIA_HOUSING_PATH: &str = "data/benchmarks/california_housing.parquet";
#[cfg(feature = "io-parquet")]
const ADULT_PATH: &str = "data/benchmarks/adult.parquet";
#[cfg(feature = "io-parquet")]
const COVERTYPE_PATH: &str = "data/benchmarks/covertype.parquet";

/// Get real-world dataset configs if parquet files exist.
#[cfg(feature = "io-parquet")]
fn real_world_configs(quick: bool) -> Vec<BenchmarkConfig> {
	let trees = if quick { 50 } else { 100 };
	let covertype_subsample = if quick { Some(10_000) } else { Some(50_000) };
	
	let mut configs = Vec::new();
	
	// California Housing (regression)
	let california_path = PathBuf::from(CALIFORNIA_HOUSING_PATH);
	if california_path.exists() {
		configs.push(BenchmarkConfig {
			name: "california_housing".to_string(),
			task: Task::Regression,
			data_source: DataSource::Parquet { path: california_path.clone(), subsample: None },
			trees,
			depth: 6,
			classes: None,
			linear_leaves: false,
		});
		// Linear GBDT variant for California Housing
		configs.push(BenchmarkConfig {
			name: "california_housing_linear".to_string(),
			task: Task::Regression,
			data_source: DataSource::Parquet { path: california_path, subsample: None },
			trees,
			depth: 6,
			classes: None,
			linear_leaves: true,
		});
	}
	
	// Adult (binary classification)
	let adult_path = PathBuf::from(ADULT_PATH);
	if adult_path.exists() {
		configs.push(BenchmarkConfig {
			name: "adult".to_string(),
			task: Task::Binary,
			data_source: DataSource::Parquet { path: adult_path, subsample: None },
			trees,
			depth: 6,
			classes: None,
			linear_leaves: false,
		});
	}
	
	// Covertype (multiclass - subsampled for speed)
	let covertype_path = PathBuf::from(COVERTYPE_PATH);
	if covertype_path.exists() {
		configs.push(BenchmarkConfig {
			name: "covertype".to_string(),
			task: Task::Multiclass,
			data_source: DataSource::Parquet { path: covertype_path, subsample: covertype_subsample },
			trees,
			depth: 6,
			classes: Some(7),
			linear_leaves: false,
		});
	}
	
	configs
}

#[cfg(not(feature = "io-parquet"))]
fn real_world_configs(_quick: bool) -> Vec<BenchmarkConfig> {
	Vec::new()
}

// =============================================================================
// Dataset Loading
// =============================================================================

fn load_libsvm_dense(path: &Path) -> (Vec<f32>, Vec<f32>, usize, usize) {
	let content = fs::read_to_string(path).expect("failed to read libsvm file");
	let mut rows: Vec<Vec<(usize, f32)>> = Vec::new();
	let mut labels: Vec<f32> = Vec::new();
	let mut max_col: usize = 0;

	for (line_idx, line) in content.lines().enumerate() {
		let line = line.trim();
		if line.is_empty() {
			continue;
		}
		let mut parts = line.split_whitespace();
		let y_str = parts.next().unwrap();
		let y: f32 = y_str.parse().unwrap_or_else(|_| panic!("invalid label at line {}", line_idx + 1));
		labels.push(y);
		let mut feats: Vec<(usize, f32)> = Vec::new();
		for p in parts {
			let (idx_str, val_str) = p
				.split_once(':')
				.unwrap_or_else(|| panic!("invalid feature token '{}' at line {}", p, line_idx + 1));
			let idx1: usize = idx_str.parse().unwrap_or_else(|_| panic!("invalid feature index at line {}", line_idx + 1));
			if idx1 == 0 {
				panic!("libsvm feature indices must be 1-based (got 0) at line {}", line_idx + 1);
			}
			let idx0 = idx1 - 1;
			let v: f32 = val_str.parse().unwrap_or_else(|_| panic!("invalid feature value at line {}", line_idx + 1));
			max_col = max_col.max(idx0 + 1);
			feats.push((idx0, v));
		}
		rows.push(feats);
	}

	let n_rows = labels.len();
	let n_cols = max_col;
	let mut x = vec![0.0f32; n_rows * n_cols];
	for (r, feats) in rows.iter().enumerate() {
		let base = r * n_cols;
		for &(c, v) in feats {
			x[base + c] = v;
		}
	}
	(x, labels, n_rows, n_cols)
}

fn load_label0_dense(path: &Path) -> (Vec<f32>, Vec<f32>, usize, usize) {
	let content = fs::read_to_string(path).expect("failed to read label0 file");
	let mut x: Vec<f32> = Vec::new();
	let mut y: Vec<f32> = Vec::new();
	let mut n_cols: Option<usize> = None;

	for (line_idx, line) in content.lines().enumerate() {
		let line = line.trim();
		if line.is_empty() {
			continue;
		}
		let parts: Vec<&str> = line.split_whitespace().collect();
		assert!(parts.len() >= 2, "expected label + at least 1 feature at line {}", line_idx + 1);
		let label: f32 = parts[0].parse().unwrap_or_else(|_| panic!("invalid label at line {}", line_idx + 1));
		y.push(label);
		let cols_here = parts.len() - 1;
		if let Some(c) = n_cols {
			assert_eq!(c, cols_here, "inconsistent column count at line {}", line_idx + 1);
		} else {
			n_cols = Some(cols_here);
		}
		for j in 0..cols_here {
			let v: f32 = parts[j + 1]
				.parse()
				.unwrap_or_else(|_| panic!("invalid feature value at line {}", line_idx + 1));
			x.push(v);
		}
	}

	let rows = y.len();
	let cols = n_cols.unwrap_or(0);
	assert_eq!(x.len(), rows * cols);
	(x, y, rows, cols)
}

fn load_uci_machine_regression(path: &Path) -> (Vec<f32>, Vec<f32>, usize, usize) {
	let content = fs::read_to_string(path).expect("failed to read UCI machine.data");
	let mut x: Vec<f32> = Vec::new();
	let mut y: Vec<f32> = Vec::new();
	for (line_idx, line) in content.lines().enumerate() {
		let line = line.trim();
		if line.is_empty() {
			continue;
		}
		let parts: Vec<&str> = line.split(',').collect();
		assert_eq!(parts.len(), 10, "expected 10 columns at line {}", line_idx + 1);
		// Columns: vendor, model, MYCT, MMIN, MMAX, CACH, CHMIN, CHMAX, PRP, ERP
		for j in 2..=7 {
			let v: f32 = parts[j]
				.parse()
				.unwrap_or_else(|_| panic!("invalid numeric feature at line {}", line_idx + 1));
			x.push(v);
		}
		let prp: f32 = parts[8]
			.parse()
			.unwrap_or_else(|_| panic!("invalid target (PRP) at line {}", line_idx + 1));
		y.push(prp);
	}
	let rows = y.len();
	let cols = 6;
	assert_eq!(x.len(), rows * cols);
	(x, y, rows, cols)
}

fn select_rows_row_major(features_row_major: &[f32], rows: usize, cols: usize, row_indices: &[usize]) -> Vec<f32> {
	assert_eq!(features_row_major.len(), rows * cols);
	let mut out = Vec::with_capacity(row_indices.len() * cols);
	for &r in row_indices {
		assert!(r < rows);
		let start = r * cols;
		out.extend_from_slice(&features_row_major[start..start + cols]);
	}
	out
}

fn select_targets(targets: &[f32], row_indices: &[usize]) -> Vec<f32> {
	let mut out = Vec::with_capacity(row_indices.len());
	for &r in row_indices {
		out.push(targets[r]);
	}
	out
}

fn extract_group(output: &booste_rs::inference::common::PredictionOutput, group: usize) -> Vec<f32> {
	let num_groups = output.num_groups();
	assert!(group < num_groups);
	if num_groups == 1 {
		return output.as_slice().to_vec();
	}
	// Column-major: column(group) is contiguous
	output.column(group).to_vec()
}

// =============================================================================
// Result structures
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunResult {
	task: String,
	seed: u64,
	booste_rs: LibraryMetrics,
	#[serde(default)]
	xgboost: Option<LibraryMetrics>,
	#[serde(default)]
	lightgbm: Option<LibraryMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct LibraryMetrics {
	#[serde(default)]
	metrics: MetricsJson,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct MetricsJson {
	#[serde(default)]
	rmse: Option<f64>,
	#[serde(default)]
	mae: Option<f64>,
	#[serde(default)]
	logloss: Option<f64>,
	#[serde(default)]
	acc: Option<f64>,
	#[serde(default)]
	mlogloss: Option<f64>,
}

#[derive(Debug, Clone)]
struct MetricStats {
	mean: f64,
	std: f64,
	#[allow(dead_code)]
	min: f64,
	#[allow(dead_code)]
	max: f64,
	n: usize,
}

impl MetricStats {
	fn from_samples(samples: &[f64]) -> Option<Self> {
		if samples.is_empty() {
			return None;
		}
		let n = samples.len();
		let mean = samples.iter().sum::<f64>() / n as f64;
		let variance = if n > 1 {
			samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
		} else {
			0.0
		};
		let std = variance.sqrt();
		let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
		let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
		Some(Self { mean, std, min, max, n })
	}

	fn format(&self, precision: usize) -> String {
		if self.n == 1 {
			format!("{:.prec$}", self.mean, prec = precision)
		} else {
			format!("{:.prec$} ± {:.prec$}", self.mean, self.std, prec = precision)
		}
	}
}

#[derive(Debug, Clone)]
struct AggregatedMetrics {
	rmse: Option<MetricStats>,
	mae: Option<MetricStats>,
	logloss: Option<MetricStats>,
	acc: Option<MetricStats>,
	mlogloss: Option<MetricStats>,
}

impl AggregatedMetrics {
	fn from_runs(runs: &[LibraryMetrics]) -> Self {
		let rmse: Vec<f64> = runs.iter().filter_map(|r| r.metrics.rmse).collect();
		let mae: Vec<f64> = runs.iter().filter_map(|r| r.metrics.mae).collect();
		let logloss: Vec<f64> = runs.iter().filter_map(|r| r.metrics.logloss).collect();
		let acc: Vec<f64> = runs.iter().filter_map(|r| r.metrics.acc).collect();
		let mlogloss: Vec<f64> = runs.iter().filter_map(|r| r.metrics.mlogloss).collect();

		Self {
			rmse: MetricStats::from_samples(&rmse),
			mae: MetricStats::from_samples(&mae),
			logloss: MetricStats::from_samples(&logloss),
			acc: MetricStats::from_samples(&acc),
			mlogloss: MetricStats::from_samples(&mlogloss),
		}
	}
}

#[derive(Debug)]
struct BenchmarkResult {
	config: BenchmarkConfig,
	boosters: AggregatedMetrics,
	xgboost: Option<AggregatedMetrics>,
	lightgbm: Option<AggregatedMetrics>,
}

// =============================================================================
// Training and Evaluation
// =============================================================================

/// Transpose from row-major to column-major format.
///
/// XGBoost/LightGBM output row-major: [row0_class0, row0_class1, ..., row1_class0, ...]
/// Our metrics expect column-major: [class0_row0, class0_row1, ..., class1_row0, ...]
#[allow(dead_code)] // Used only with bench-xgboost/bench-lightgbm features
fn transpose_row_to_col_major(row_major: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32> {
	debug_assert_eq!(row_major.len(), n_rows * n_cols);
	let mut col_major = vec![0.0f32; n_rows * n_cols];
	for row in 0..n_rows {
		for col in 0..n_cols {
			col_major[col * n_rows + row] = row_major[row * n_cols + col];
		}
	}
	col_major
}

fn load_data(
	config: &BenchmarkConfig,
	seed: u64,
	valid_fraction: f32,
) -> (Vec<f32>, Vec<f32>, usize, Vec<f32>, Vec<f32>, usize, usize) {
	match &config.data_source {
		DataSource::Synthetic { rows, cols } => {
			let x = random_dense_f32(*rows, *cols, seed, -1.0, 1.0);
			let y = match config.task {
				Task::Regression => synthetic_regression_targets_linear(&x, *rows, *cols, seed ^ 0x0BAD_5EED, 0.05).0,
				Task::Binary => synthetic_binary_targets_from_linear_score(&x, *rows, *cols, seed ^ 0xB1A2_0001, 0.2),
				Task::Multiclass => {
					let num_classes = config.classes.unwrap_or(3);
					synthetic_multiclass_targets_from_linear_scores(&x, *rows, *cols, num_classes, seed ^ 0x00C1_A550, 0.1)
				}
			};
			let (train_idx, valid_idx) = split_indices(*rows, valid_fraction, seed ^ 0x51EED);
			let x_train = select_rows_row_major(&x, *rows, *cols, &train_idx);
			let y_train = select_targets(&y, &train_idx);
			let x_valid = select_rows_row_major(&x, *rows, *cols, &valid_idx);
			let y_valid = select_targets(&y, &valid_idx);
			(x_train, y_train, train_idx.len(), x_valid, y_valid, valid_idx.len(), *cols)
		}
		#[cfg(feature = "io-parquet")]
		DataSource::Parquet { path, subsample } => {
			let (x_all, y_all, rows_orig, cols) = load_parquet_xy_row_major_f32(path).expect("failed to load parquet");
			
			// Optionally subsample large datasets
			let (x, y, rows) = if let Some(max_rows) = subsample {
				if rows_orig > *max_rows {
					// Deterministic subsampling based on seed
					use rand::prelude::*;
					use rand::SeedableRng;
					let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed ^ 0x5B5A_AAAA);
					let mut indices: Vec<usize> = (0..rows_orig).collect();
					indices.shuffle(&mut rng);
					indices.truncate(*max_rows);
					indices.sort_unstable();
					
					let x_sub = select_rows_row_major(&x_all, rows_orig, cols, &indices);
					let y_sub = select_targets(&y_all, &indices);
					(x_sub, y_sub, *max_rows)
				} else {
					(x_all, y_all, rows_orig)
				}
			} else {
				(x_all, y_all, rows_orig)
			};
			
			let (train_idx, valid_idx) = split_indices(rows, valid_fraction, seed ^ 0x51EED);
			let x_train = select_rows_row_major(&x, rows, cols, &train_idx);
			let y_train = select_targets(&y, &train_idx);
			let x_valid = select_rows_row_major(&x, rows, cols, &valid_idx);
			let y_valid = select_targets(&y, &valid_idx);
			(x_train, y_train, train_idx.len(), x_valid, y_valid, valid_idx.len(), cols)
		}
		DataSource::LibSvm(path) => {
			let (x, y, rows, cols) = load_libsvm_dense(path);
			let (train_idx, valid_idx) = split_indices(rows, valid_fraction, seed ^ 0x51EED);
			let x_train = select_rows_row_major(&x, rows, cols, &train_idx);
			let y_train = select_targets(&y, &train_idx);
			let x_valid = select_rows_row_major(&x, rows, cols, &valid_idx);
			let y_valid = select_targets(&y, &valid_idx);
			(x_train, y_train, train_idx.len(), x_valid, y_valid, valid_idx.len(), cols)
		}
		DataSource::UciMachine(path) => {
			let (x, y, rows, cols) = load_uci_machine_regression(path);
			let (train_idx, valid_idx) = split_indices(rows, valid_fraction, seed ^ 0x51EED);
			let x_train = select_rows_row_major(&x, rows, cols, &train_idx);
			let y_train = select_targets(&y, &train_idx);
			let x_valid = select_rows_row_major(&x, rows, cols, &valid_idx);
			let y_valid = select_targets(&y, &valid_idx);
			(x_train, y_train, train_idx.len(), x_valid, y_valid, valid_idx.len(), cols)
		}
		DataSource::Label0(path) => {
			let (x, y, rows, cols) = load_label0_dense(path);
			let (train_idx, valid_idx) = split_indices(rows, valid_fraction, seed ^ 0x51EED);
			let x_train = select_rows_row_major(&x, rows, cols, &train_idx);
			let y_train = select_targets(&y, &train_idx);
			let x_valid = select_rows_row_major(&x, rows, cols, &valid_idx);
			let y_valid = select_targets(&y, &valid_idx);
			(x_train, y_train, train_idx.len(), x_valid, y_valid, valid_idx.len(), cols)
		}
	}
}

fn train_boosters(
	config: &BenchmarkConfig,
	x_train: &[f32],
	y_train: &[f32],
	rows_train: usize,
	x_valid: &[f32],
	y_valid: &[f32],
	rows_valid: usize,
	cols: usize,
	seed: u64,
) -> LibraryMetrics {
	let row_train: DenseMatrix<f32, RowMajor> = DenseMatrix::from_vec(x_train.to_vec(), rows_train, cols);
	let col_train: ColMatrix<f32> = row_train.to_layout();
	let binned_train = BinnedDatasetBuilder::from_matrix(&col_train, 256).build().unwrap();
	let row_valid: RowMatrix<f32> = RowMatrix::from_vec(x_valid.to_vec(), rows_valid, cols);

	let linear_leaves = if config.linear_leaves {
		Some(LinearLeafConfig::default())
	} else {
		None
	};

	let params = GBDTParams {
		n_trees: config.trees,
		learning_rate: 0.1,
		growth_strategy: GrowthStrategy::DepthWise { max_depth: config.depth },
		gain: GainParams { reg_lambda: 1.0, ..Default::default() },
		n_threads: 1,
		cache_size: 256,
		seed,
		linear_leaves,
		..Default::default()
	};

	match config.task {
		Task::Regression => {
			let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);
			let forest = trainer.train(&binned_train, y_train, &[], &[]).unwrap();
			let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);
			let pred = predictor.predict(&row_valid);
			let pred0 = extract_group(&pred, 0);
			let rmse = Rmse.compute(rows_valid, 1, &pred0, y_valid, &[]);
			let mae = Mae.compute(rows_valid, 1, &pred0, y_valid, &[]);
			LibraryMetrics { metrics: MetricsJson { rmse: Some(rmse), mae: Some(mae), ..Default::default() } }
		}
		Task::Binary => {
			let objective = LogisticLoss;
			let trainer = GBDTTrainer::new(objective, LogLoss, params);
			let forest = trainer.train(&binned_train, y_train, &[], &[]).unwrap();
			let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);
			let mut raw = predictor.predict(&row_valid);
			objective.transform_prediction_inplace(&mut raw);
			let prob = raw.column(0);
			let ll = LogLoss.compute(rows_valid, 1, prob, y_valid, &[]);
			let acc = Accuracy::default().compute(rows_valid, 1, prob, y_valid, &[]);
			LibraryMetrics { metrics: MetricsJson { logloss: Some(ll), acc: Some(acc), ..Default::default() } }
		}
		Task::Multiclass => {
			let num_classes = config.classes.unwrap_or(3);
			let objective = SoftmaxLoss::new(num_classes);
			let trainer = GBDTTrainer::new(objective, MulticlassLogLoss, params);
			let forest = trainer.train(&binned_train, y_train, &[], &[]).unwrap();
			let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);
			let mut raw = predictor.predict(&row_valid);
			// Apply softmax to column-major output
			objective.transform_prediction_inplace(&mut raw);
			// Metrics can now use column-major directly
			let prob_col_major = raw.as_slice();
			let ll = MulticlassLogLoss.compute(rows_valid, num_classes, prob_col_major, y_valid, &[]);
			let acc = MulticlassAccuracy.compute(rows_valid, num_classes, prob_col_major, y_valid, &[]);
			LibraryMetrics { metrics: MetricsJson { mlogloss: Some(ll), acc: Some(acc), ..Default::default() } }
		}
	}
}

#[cfg(feature = "bench-xgboost")]
fn train_xgboost(
	config: &BenchmarkConfig,
	x_train: &[f32],
	y_train: &[f32],
	rows_train: usize,
	x_valid: &[f32],
	y_valid: &[f32],
	rows_valid: usize,
	_cols: usize,
) -> LibraryMetrics {
	let tree_params = TreeBoosterParametersBuilder::default()
		.eta(0.1)
		.max_depth(config.depth)
		.max_leaves(0u32)
		.grow_policy(GrowPolicy::Depthwise)
		.lambda(1.0)
		.alpha(0.0)
		.gamma(0.0)
		.min_child_weight(1.0)
		.tree_method(TreeMethod::Hist)
		.max_bin(256u32)
		.build()
		.unwrap();

	let num_classes = config.classes.unwrap_or(3);
	let xgb_objective = match config.task {
		Task::Regression => XgbObjective::RegLinear,
		Task::Binary => XgbObjective::BinaryLogistic,
		Task::Multiclass => XgbObjective::MultiSoftprob(num_classes as u32),
	};
	let learning_params = LearningTaskParametersBuilder::default().objective(xgb_objective).build().unwrap();

	let booster_params = BoosterParametersBuilder::default()
		.booster_type(BoosterType::Tree(tree_params))
		.learning_params(learning_params)
		.verbose(false)
		.threads(Some(1))
		.build()
		.unwrap();

	let x_train_f32: Vec<f32> = x_train.to_vec();
	let mut dtrain = DMatrix::from_dense(&x_train_f32, rows_train).unwrap();
	dtrain.set_labels(y_train).unwrap();
	let x_valid_f32: Vec<f32> = x_valid.to_vec();
	let mut dvalid = DMatrix::from_dense(&x_valid_f32, rows_valid).unwrap();
	dvalid.set_labels(y_valid).unwrap();

	let training_params = TrainingParametersBuilder::default()
		.dtrain(&dtrain)
		.boost_rounds(config.trees)
		.booster_params(booster_params)
		.evaluation_sets(None)
		.build()
		.unwrap();
	let model = XgbBooster::train(&training_params).unwrap();
	let pred = model.predict(&dvalid).unwrap();

	match config.task {
		Task::Regression => {
			let pred_f32: Vec<f32> = pred.into_iter().map(|x| x as f32).collect();
			let rmse = Rmse.compute(rows_valid, 1, &pred_f32, y_valid, &[]);
			let mae = Mae.compute(rows_valid, 1, &pred_f32, y_valid, &[]);
			LibraryMetrics { metrics: MetricsJson { rmse: Some(rmse), mae: Some(mae), ..Default::default() } }
		}
		Task::Binary => {
			let pred_f32: Vec<f32> = pred.into_iter().map(|x| x as f32).collect();
			let ll = LogLoss.compute(rows_valid, 1, &pred_f32, y_valid, &[]);
			let acc = Accuracy::default().compute(rows_valid, 1, &pred_f32, y_valid, &[]);
			LibraryMetrics { metrics: MetricsJson { logloss: Some(ll), acc: Some(acc), ..Default::default() } }
		}
		Task::Multiclass => {
			// XGBoost outputs row-major: [row0_class0, row0_class1, ..., row1_class0, ...]
			// Our metrics expect column-major: [class0_row0, class0_row1, ..., class1_row0, ...]
			let prob_row_major: Vec<f32> = pred.into_iter().map(|x| x as f32).collect();
			let prob_col_major = transpose_row_to_col_major(&prob_row_major, rows_valid, num_classes);
			let ll = MulticlassLogLoss.compute(rows_valid, num_classes, &prob_col_major, y_valid, &[]);
			let acc = MulticlassAccuracy.compute(rows_valid, num_classes, &prob_col_major, y_valid, &[]);
			LibraryMetrics { metrics: MetricsJson { mlogloss: Some(ll), acc: Some(acc), ..Default::default() } }
		}
	}
}

#[cfg(feature = "bench-lightgbm")]
fn train_lightgbm(
	config: &BenchmarkConfig,
	x_train: &[f32],
	y_train: &[f32],
	_rows_train: usize,
	x_valid: &[f32],
	y_valid: &[f32],
	rows_valid: usize,
	cols: usize,
) -> LibraryMetrics {
	let x_train_f64: Vec<f64> = x_train.iter().map(|&x| x as f64).collect();
	let x_valid_f64: Vec<f64> = x_valid.iter().map(|&x| x as f64).collect();

	let num_classes = config.classes.unwrap_or(3);
	let mut params = match config.task {
		Task::Regression => serde_json::json!({"objective":"regression","metric":"l2"}),
		Task::Binary => serde_json::json!({"objective":"binary","metric":"binary_logloss"}),
		Task::Multiclass => serde_json::json!({"objective":"multiclass","metric":"multi_logloss","num_class": num_classes}),
	};
	params["num_iterations"] = serde_json::Value::from(config.trees as i64);
	params["learning_rate"] = serde_json::Value::from(0.1f64);
	params["max_bin"] = serde_json::Value::from(256i64);
	params["max_depth"] = serde_json::Value::from(config.depth as i64);
	let num_leaves = (1u64.checked_shl(config.depth).unwrap_or(u64::MAX)).max(2) as i64;
	params["num_leaves"] = serde_json::Value::from(num_leaves);
	params["min_data_in_leaf"] = serde_json::Value::from(1i64);
	params["lambda_l2"] = serde_json::Value::from(1.0f64);
	params["feature_fraction"] = serde_json::Value::from(1.0f64);
	params["bagging_fraction"] = serde_json::Value::from(1.0f64);
	params["bagging_freq"] = serde_json::Value::from(0i64);
	params["verbosity"] = serde_json::Value::from(-1i64);
	params["num_threads"] = serde_json::Value::from(1i64);
	// Enable linear trees if configured
	if config.linear_leaves {
		params["linear_tree"] = serde_json::Value::from(true);
	}

	let ds = lightgbm3::Dataset::from_slice(&x_train_f64, y_train, cols as i32, true).unwrap();
	let bst = lightgbm3::Booster::train(ds, &params).unwrap();
	let pred = bst.predict(&x_valid_f64, cols as i32, true).unwrap();

	match config.task {
		Task::Regression => {
			let pred_f32: Vec<f32> = pred.into_iter().map(|x| x as f32).collect();
			let rmse = Rmse.compute(rows_valid, 1, &pred_f32, y_valid, &[]);
			let mae = Mae.compute(rows_valid, 1, &pred_f32, y_valid, &[]);
			LibraryMetrics { metrics: MetricsJson { rmse: Some(rmse), mae: Some(mae), ..Default::default() } }
		}
		Task::Binary => {
			let pred_f32: Vec<f32> = pred.into_iter().map(|x| x as f32).collect();
			let ll = LogLoss.compute(rows_valid, 1, &pred_f32, y_valid, &[]);
			let acc = Accuracy::default().compute(rows_valid, 1, &pred_f32, y_valid, &[]);
			LibraryMetrics { metrics: MetricsJson { logloss: Some(ll), acc: Some(acc), ..Default::default() } }
		}
		Task::Multiclass => {
			// LightGBM outputs row-major: [row0_class0, row0_class1, ..., row1_class0, ...]
			// Our metrics expect column-major: [class0_row0, class0_row1, ..., class1_row0, ...]
			let prob_row_major: Vec<f32> = pred.into_iter().map(|x| x as f32).collect();
			let prob_col_major = transpose_row_to_col_major(&prob_row_major, rows_valid, num_classes);
			let ll = MulticlassLogLoss.compute(rows_valid, num_classes, &prob_col_major, y_valid, &[]);
			let acc = MulticlassAccuracy.compute(rows_valid, num_classes, &prob_col_major, y_valid, &[]);
			LibraryMetrics { metrics: MetricsJson { mlogloss: Some(ll), acc: Some(acc), ..Default::default() } }
		}
	}
}

fn run_single_eval(config: &BenchmarkConfig, seed: u64) -> RunResult {
	let (x_train, y_train, rows_train, x_valid, y_valid, rows_valid, cols) = load_data(config, seed, 0.2);

	let booste_rs = train_boosters(config, &x_train, &y_train, rows_train, &x_valid, &y_valid, rows_valid, cols, seed);

	// XGBoost doesn't support linear trees, so skip it for linear tree configs
	#[cfg(feature = "bench-xgboost")]
	let xgboost = if config.linear_leaves {
		None
	} else {
		Some(train_xgboost(config, &x_train, &y_train, rows_train, &x_valid, &y_valid, rows_valid, cols))
	};
	#[cfg(not(feature = "bench-xgboost"))]
	let xgboost: Option<LibraryMetrics> = None;

	// LightGBM linear_tree support requires specific library version
	// The lightgbm3 crate may not support it properly, so skip for linear configs
	#[cfg(feature = "bench-lightgbm")]
	let lightgbm = if config.linear_leaves {
		None // Skip - lightgbm3 crate crashes with linear_tree=true
	} else {
		Some(train_lightgbm(config, &x_train, &y_train, rows_train, &x_valid, &y_valid, rows_valid, cols))
	};
	#[cfg(not(feature = "bench-lightgbm"))]
	let lightgbm: Option<LibraryMetrics> = None;

	RunResult { task: config.task.as_str().to_string(), seed, booste_rs, xgboost, lightgbm }
}

fn run_benchmark(config: &BenchmarkConfig, seeds: &[u64]) -> BenchmarkResult {
	println!("Running benchmark: {} ({} seeds)", config.name, seeds.len());

	let mut boosters_runs: Vec<LibraryMetrics> = Vec::new();
	let mut xgb_runs: Vec<LibraryMetrics> = Vec::new();
	let mut lgb_runs: Vec<LibraryMetrics> = Vec::new();

	for (i, &seed) in seeds.iter().enumerate() {
		print!("  Seed {}/{}: {} ... ", i + 1, seeds.len(), seed);
		std::io::stdout().flush().unwrap();

		let result = run_single_eval(config, seed);
		boosters_runs.push(result.booste_rs);
		if let Some(xgb) = result.xgboost {
			xgb_runs.push(xgb);
		}
		if let Some(lgb) = result.lightgbm {
			lgb_runs.push(lgb);
		}
		println!("OK");
	}

	BenchmarkResult {
		config: config.clone(),
		boosters: AggregatedMetrics::from_runs(&boosters_runs),
		xgboost: if xgb_runs.is_empty() { None } else { Some(AggregatedMetrics::from_runs(&xgb_runs)) },
		lightgbm: if lgb_runs.is_empty() { None } else { Some(AggregatedMetrics::from_runs(&lgb_runs)) },
	}
}

// =============================================================================
// Report generation
// =============================================================================

fn find_best(vals: &[Option<&MetricStats>], lower_is_better: bool) -> Option<usize> {
	let valids: Vec<(usize, f64)> = vals.iter().enumerate().filter_map(|(i, opt)| opt.map(|s| (i, s.mean))).collect();

	if valids.is_empty() {
		return None;
	}

	let best = if lower_is_better {
		valids.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
	} else {
		valids.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
	};

	best.map(|(i, _)| *i)
}

fn format_cell(stats: Option<&MetricStats>, precision: usize, is_best: bool) -> String {
	match stats {
		Some(s) => {
			let val = s.format(precision);
			if is_best { format!("**{}**", val) } else { val }
		}
		None => "-".to_string(),
	}
}

fn generate_report(results: &[BenchmarkResult], seeds: &[u64]) -> String {
	let mut out = String::new();

	out.push_str("# Quality Benchmark Report\n\n");
	out.push_str(&format!("**Seeds**: {} ({:?})\n\n", seeds.len(), seeds));

	// Group results by task
	let mut by_task: HashMap<&str, Vec<&BenchmarkResult>> = HashMap::new();
	for r in results {
		by_task.entry(r.config.task.as_str()).or_default().push(r);
	}

	// Ensure consistent ordering
	for task in ["regression", "binary", "multiclass"] {
		let Some(task_results) = by_task.get(task) else { continue };

		out.push_str(&format!("## {}\n\n", task.to_uppercase()));

		match task {
			"regression" => {
				out.push_str("### RMSE (lower is better)\n\n");
				out.push_str("| Dataset | booste-rs | XGBoost | LightGBM |\n");
				out.push_str("|---------|-----------|---------|----------|\n");

				for r in task_results {
					let boosters = r.boosters.rmse.as_ref();
					let xgb = r.xgboost.as_ref().and_then(|x| x.rmse.as_ref());
					let lgb = r.lightgbm.as_ref().and_then(|x| x.rmse.as_ref());
					let best = find_best(&[boosters, xgb, lgb], true);

					out.push_str(&format!(
						"| {} | {} | {} | {} |\n",
						r.config.name,
						format_cell(boosters, 6, best == Some(0)),
						format_cell(xgb, 6, best == Some(1)),
						format_cell(lgb, 6, best == Some(2)),
					));
				}
				out.push_str("\n");

				out.push_str("### MAE (lower is better)\n\n");
				out.push_str("| Dataset | booste-rs | XGBoost | LightGBM |\n");
				out.push_str("|---------|-----------|---------|----------|\n");

				for r in task_results {
					let boosters = r.boosters.mae.as_ref();
					let xgb = r.xgboost.as_ref().and_then(|x| x.mae.as_ref());
					let lgb = r.lightgbm.as_ref().and_then(|x| x.mae.as_ref());
					let best = find_best(&[boosters, xgb, lgb], true);

					out.push_str(&format!(
						"| {} | {} | {} | {} |\n",
						r.config.name,
						format_cell(boosters, 6, best == Some(0)),
						format_cell(xgb, 6, best == Some(1)),
						format_cell(lgb, 6, best == Some(2)),
					));
				}
				out.push_str("\n");
			}
			"binary" => {
				out.push_str("### LogLoss (lower is better)\n\n");
				out.push_str("| Dataset | booste-rs | XGBoost | LightGBM |\n");
				out.push_str("|---------|-----------|---------|----------|\n");

				for r in task_results {
					let boosters = r.boosters.logloss.as_ref();
					let xgb = r.xgboost.as_ref().and_then(|x| x.logloss.as_ref());
					let lgb = r.lightgbm.as_ref().and_then(|x| x.logloss.as_ref());
					let best = find_best(&[boosters, xgb, lgb], true);

					out.push_str(&format!(
						"| {} | {} | {} | {} |\n",
						r.config.name,
						format_cell(boosters, 6, best == Some(0)),
						format_cell(xgb, 6, best == Some(1)),
						format_cell(lgb, 6, best == Some(2)),
					));
				}
				out.push_str("\n");

				out.push_str("### Accuracy (higher is better)\n\n");
				out.push_str("| Dataset | booste-rs | XGBoost | LightGBM |\n");
				out.push_str("|---------|-----------|---------|----------|\n");

				for r in task_results {
					let boosters = r.boosters.acc.as_ref();
					let xgb = r.xgboost.as_ref().and_then(|x| x.acc.as_ref());
					let lgb = r.lightgbm.as_ref().and_then(|x| x.acc.as_ref());
					let best = find_best(&[boosters, xgb, lgb], false);

					out.push_str(&format!(
						"| {} | {} | {} | {} |\n",
						r.config.name,
						format_cell(boosters, 4, best == Some(0)),
						format_cell(xgb, 4, best == Some(1)),
						format_cell(lgb, 4, best == Some(2)),
					));
				}
				out.push_str("\n");
			}
			"multiclass" => {
				out.push_str("### Multi-class LogLoss (lower is better)\n\n");
				out.push_str("| Dataset | booste-rs | XGBoost | LightGBM |\n");
				out.push_str("|---------|-----------|---------|----------|\n");

				for r in task_results {
					let boosters = r.boosters.mlogloss.as_ref();
					let xgb = r.xgboost.as_ref().and_then(|x| x.mlogloss.as_ref());
					let lgb = r.lightgbm.as_ref().and_then(|x| x.mlogloss.as_ref());
					let best = find_best(&[boosters, xgb, lgb], true);

					out.push_str(&format!(
						"| {} | {} | {} | {} |\n",
						r.config.name,
						format_cell(boosters, 6, best == Some(0)),
						format_cell(xgb, 6, best == Some(1)),
						format_cell(lgb, 6, best == Some(2)),
					));
				}
				out.push_str("\n");

				out.push_str("### Accuracy (higher is better)\n\n");
				out.push_str("| Dataset | booste-rs | XGBoost | LightGBM |\n");
				out.push_str("|---------|-----------|---------|----------|\n");

				for r in task_results {
					let boosters = r.boosters.acc.as_ref();
					let xgb = r.xgboost.as_ref().and_then(|x| x.acc.as_ref());
					let lgb = r.lightgbm.as_ref().and_then(|x| x.acc.as_ref());
					let best = find_best(&[boosters, xgb, lgb], false);

					out.push_str(&format!(
						"| {} | {} | {} | {} |\n",
						r.config.name,
						format_cell(boosters, 4, best == Some(0)),
						format_cell(xgb, 4, best == Some(1)),
						format_cell(lgb, 4, best == Some(2)),
					));
				}
				out.push_str("\n");
			}
			_ => {}
		}
	}

	out.push_str("## Benchmark Configuration\n\n");
	out.push_str("| Dataset | Data Source | Trees | Depth | Classes | Linear |\n");
	out.push_str("|---------|-------------|-------|-------|---------|--------|\n");
	for r in results {
		let source = match &r.config.data_source {
			DataSource::Synthetic { rows, cols } => format!("Synthetic {}x{}", rows, cols),
			#[cfg(feature = "io-parquet")]
			DataSource::Parquet { path, subsample } => {
				let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "parquet".to_string());
				if let Some(n) = subsample {
					format!("{} (subsampled to {})", name, n)
				} else {
					name
				}
			}
			DataSource::LibSvm(p) => format!("libsvm: {}", p.display()),
			DataSource::UciMachine(p) => format!("uci-machine: {}", p.display()),
			DataSource::Label0(p) => format!("label0: {}", p.display()),
		};
		let linear_str = if r.config.linear_leaves { "✓" } else { "-" };
		out.push_str(&format!(
			"| {} | {} | {} | {} | {} | {} |\n",
			r.config.name,
			source,
			r.config.trees,
			r.config.depth,
			r.config.classes.map(|c| c.to_string()).unwrap_or("-".to_string()),
			linear_str,
		));
	}
	out.push_str("\n");

	out
}

// =============================================================================
// Main
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchmarkMode {
	All,
	Synthetic,
	Real,
}

struct Args {
	num_seeds: usize,
	out: Option<PathBuf>,
	quick: bool,
	mode: BenchmarkMode,
	libsvm_paths: Vec<PathBuf>,
	uci_machine_paths: Vec<PathBuf>,
	label0_paths: Vec<PathBuf>,
}

fn parse_args() -> Args {
	let mut num_seeds = 5usize;
	let mut out: Option<PathBuf> = None;
	let mut quick = false;
	let mut mode = BenchmarkMode::All;
	let mut libsvm_paths: Vec<PathBuf> = Vec::new();
	let mut uci_machine_paths: Vec<PathBuf> = Vec::new();
	let mut label0_paths: Vec<PathBuf> = Vec::new();

	let mut it = std::env::args().skip(1);
	while let Some(arg) = it.next() {
		match arg.as_str() {
			"--seeds" => num_seeds = it.next().expect("--seeds value").parse().unwrap(),
			"--out" => out = Some(PathBuf::from(it.next().expect("--out path"))),
			"--quick" => quick = true,
			"--mode" => {
				let val = it.next().expect("--mode value");
				mode = match val.as_str() {
					"all" => BenchmarkMode::All,
					"synthetic" => BenchmarkMode::Synthetic,
					"real" => BenchmarkMode::Real,
					other => panic!("invalid mode: {other} (expected: all, synthetic, real)"),
				};
			}
			// Legacy flag, kept for backwards compatibility
			"--no-real" => mode = BenchmarkMode::Synthetic,
			"--libsvm" => libsvm_paths.push(PathBuf::from(it.next().expect("--libsvm path"))),
			"--uci-machine" => uci_machine_paths.push(PathBuf::from(it.next().expect("--uci-machine path"))),
			"--label0" => label0_paths.push(PathBuf::from(it.next().expect("--label0 path"))),
			"--help" => {
				eprintln!(
					"quality_benchmark\n\n  --seeds <n>         Number of seeds (default: 5)\n  --out <path>        Output markdown file\n  --quick             Quick mode (fewer rows/trees)\n  --mode <mode>       Benchmark mode: all (default), synthetic, real\n  --no-real           Alias for --mode synthetic\n  --libsvm <path>     Add libsvm regression dataset\n  --uci-machine <path> Add UCI machine.data dataset\n  --label0 <path>     Add label0 regression dataset"
				);
				std::process::exit(0);
			}
			other => panic!("unknown arg: {other}"),
		}
	}

	Args { num_seeds, out, quick, mode, libsvm_paths, uci_machine_paths, label0_paths }
}

fn main() {
	let args = parse_args();

	// Generate seeds
	let seeds: Vec<u64> = (0..args.num_seeds).map(|i| 42 + i as u64 * 1337).collect();

	let mut configs = Vec::new();
	
	// Add synthetic datasets if mode is All or Synthetic
	if args.mode == BenchmarkMode::All || args.mode == BenchmarkMode::Synthetic {
		configs.extend(default_configs(args.quick));
	}
	
	// Add real-world datasets if mode is All or Real
	if args.mode == BenchmarkMode::All || args.mode == BenchmarkMode::Real {
		let real_configs = real_world_configs(args.quick);
		if !real_configs.is_empty() {
			println!("Found {} real-world dataset(s)", real_configs.len());
			configs.extend(real_configs);
		} else if args.mode == BenchmarkMode::Real {
			eprintln!("Warning: --mode real specified but no real-world datasets found.");
			eprintln!("Generate them with: cd tools/data_generation && uv run python scripts/generate_benchmark_datasets.py");
			eprintln!("Make sure to compile with --features io-parquet");
		}
	}

	// Add user-specified datasets
	let trees = if args.quick { 50 } else { 100 };
	for (i, path) in args.libsvm_paths.iter().enumerate() {
		let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| format!("libsvm_{}", i));
		configs.push(BenchmarkConfig {
			name: format!("libsvm_{}", name),
			task: Task::Regression,
			data_source: DataSource::LibSvm(path.clone()),
			trees,
			depth: 6,
			classes: None,
			linear_leaves: false,
		});
	}
	for (i, path) in args.uci_machine_paths.iter().enumerate() {
		let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| format!("uci_{}", i));
		configs.push(BenchmarkConfig {
			name: format!("uci_{}", name),
			task: Task::Regression,
			data_source: DataSource::UciMachine(path.clone()),
			trees,
			depth: 6,
			classes: None,
			linear_leaves: false,
		});
	}
	for (i, path) in args.label0_paths.iter().enumerate() {
		let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| format!("label0_{}", i));
		configs.push(BenchmarkConfig {
			name: format!("label0_{}", name),
			task: Task::Regression,
			data_source: DataSource::Label0(path.clone()),
			trees,
			depth: 6,
			classes: None,
			linear_leaves: false,
		});
	}

	println!("=== Quality Benchmark ===");
	println!("Running {} configurations with {} seeds each", configs.len(), seeds.len());
	println!();

	let mut results: Vec<BenchmarkResult> = Vec::new();
	for config in &configs {
		results.push(run_benchmark(config, &seeds));
	}

	let report = generate_report(&results, &seeds);

	if let Some(path) = args.out {
		fs::write(&path, &report).expect("failed to write report");
		println!("\nReport written to: {}", path.display());
	} else {
		println!("\n{}", report);
	}
}
