//! Quality evaluation harness (not run in CI).
//!
//! This is intended to complement Criterion performance benchmarks with
//! simple, reproducible model-quality comparisons across libraries.
//!
//! Examples:
//! - Synthetic regression (booste-rs only):
//!   `cargo run --bin quality_eval --release -- --task regression --synthetic 50000 100 --trees 200 --depth 6`
//!
//! - Arrow IPC dataset (requires io-arrow):
//!   `cargo run --bin quality_eval --release --features io-arrow -- --task regression --ipc data.arrow --trees 200 --depth 6`
//!
//! - Parquet dataset (requires io-parquet):
//!   `cargo run --bin quality_eval --release --features io-parquet -- --task regression --parquet data.parquet --trees 200 --depth 6`

use std::fs;
use std::path::{Path, PathBuf};

use booste_rs::data::{binned::BinnedDatasetBuilder, ColMatrix, DenseMatrix, RowMajor, RowMatrix};
use booste_rs::inference::common::{sigmoid_inplace, softmax_inplace};
use booste_rs::inference::gbdt::{Predictor, UnrolledTraversal6};
use booste_rs::testing::data::{
	random_dense_f32, split_indices, synthetic_binary_targets_from_linear_score,
	synthetic_multiclass_targets_from_linear_scores, synthetic_regression_targets_linear,
};
use booste_rs::training::{GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, LogisticLoss, SoftmaxLoss, SquaredLoss};

use booste_rs::training::{Accuracy, LogLoss, Mae, Metric, MulticlassAccuracy, MulticlassLogLoss, Rmse};

use serde::{Deserialize, Serialize};

#[cfg(feature = "io-arrow")]
use booste_rs::data::io::arrow::load_ipc_xy_row_major_f32;
#[cfg(feature = "io-parquet")]
use booste_rs::data::io::parquet::load_parquet_xy_row_major_f32;

#[cfg(feature = "bench-xgboost")]
use xgb::parameters::tree::{GrowPolicy, TreeBoosterParametersBuilder, TreeMethod};
#[cfg(feature = "bench-xgboost")]
use xgb::parameters::{learning::LearningTaskParametersBuilder, learning::Objective, BoosterParametersBuilder, TrainingParametersBuilder};
#[cfg(feature = "bench-xgboost")]
use xgb::parameters::BoosterType;
#[cfg(feature = "bench-xgboost")]
use xgb::{Booster as XgbBooster, DMatrix};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Task {
	Regression,
	Binary,
	Multiclass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Growth {
	DepthWise,
	LeafWise,
}

#[derive(Debug)]
struct Args {
	task: Task,
	rows: usize,
	cols: usize,
	num_classes: usize,
	growth: Growth,
	depth: u32,
	max_leaves: u32,
	trees: u32,
	seed: u64,
	valid_fraction: f32,
	ipc: Option<PathBuf>,
	parquet: Option<PathBuf>,
	libsvm: Option<PathBuf>,
	uci_machine: Option<PathBuf>,
	label0: Option<PathBuf>,
	xgb_gbtree_case: Option<String>,
	out: Option<PathBuf>,
	out_json: Option<PathBuf>,
}

fn parse_args() -> Args {
	let mut task = Task::Regression;
	let mut rows = 50_000usize;
	let mut cols = 100usize;
	let mut num_classes = 3usize;
	let mut growth = Growth::DepthWise;
	let mut depth = 6u32;
	let mut max_leaves: u32 = 0;
	let mut trees = 200u32;
	let mut seed = 42u64;
	let mut valid_fraction = 0.2f32;
	let mut ipc: Option<PathBuf> = None;
	let mut parquet: Option<PathBuf> = None;
	let mut libsvm: Option<PathBuf> = None;
	let mut uci_machine: Option<PathBuf> = None;
	let mut label0: Option<PathBuf> = None;
	let mut xgb_gbtree_case: Option<String> = None;
	let mut out: Option<PathBuf> = None;
	let mut out_json: Option<PathBuf> = None;

	let mut it = std::env::args().skip(1);
	while let Some(arg) = it.next() {
		match arg.as_str() {
			"--task" => {
				let v = it.next().expect("--task requires a value");
				task = match v.as_str() {
					"regression" => Task::Regression,
					"binary" => Task::Binary,
					"multiclass" => Task::Multiclass,
					_ => panic!("unknown task: {v}"),
				};
			}
			"--growth" => {
				let v = it.next().expect("--growth requires a value");
				growth = match v.as_str() {
					"depthwise" => Growth::DepthWise,
					"leafwise" => Growth::LeafWise,
					_ => panic!("unknown growth: {v} (expected depthwise|leafwise)"),
				};
			}
			"--synthetic" => {
				rows = it.next().expect("--synthetic rows").parse().unwrap();
				cols = it.next().expect("--synthetic cols").parse().unwrap();
			}
			"--classes" => num_classes = it.next().expect("--classes value").parse().unwrap(),
			"--depth" => depth = it.next().expect("--depth value").parse().unwrap(),
			"--leaves" => max_leaves = it.next().expect("--leaves value").parse().unwrap(),
			"--trees" => trees = it.next().expect("--trees value").parse().unwrap(),
			"--seed" => seed = it.next().expect("--seed value").parse().unwrap(),
			"--valid" => valid_fraction = it.next().expect("--valid value").parse().unwrap(),
			"--ipc" => ipc = Some(PathBuf::from(it.next().expect("--ipc path"))),
			"--parquet" => parquet = Some(PathBuf::from(it.next().expect("--parquet path"))),
			"--libsvm" => libsvm = Some(PathBuf::from(it.next().expect("--libsvm path"))),
			"--uci-machine" => uci_machine = Some(PathBuf::from(it.next().expect("--uci-machine path"))),
			"--label0" => label0 = Some(PathBuf::from(it.next().expect("--label0 path"))),
			"--xgb-gbtree-case" => xgb_gbtree_case = Some(it.next().expect("--xgb-gbtree-case name")),
			"--out" => out = Some(PathBuf::from(it.next().expect("--out path"))),
			"--out-json" => out_json = Some(PathBuf::from(it.next().expect("--out-json path"))),
			"--help" => {
				print_help_and_exit();
			}
			other => panic!("unknown arg: {other}"),
		}
	}

	let data_sources = [
		ipc.is_some(),
		parquet.is_some(),
		libsvm.is_some(),
		uci_machine.is_some(),
		label0.is_some(),
		xgb_gbtree_case.is_some(),
	]
		.into_iter()
		.filter(|x| *x)
		.count();
	if data_sources > 1 {
		panic!(
			"Please provide only one dataset source: --synthetic / --ipc / --parquet / --libsvm / --uci-machine / --label0 / --xgb-gbtree-case"
		);
	}

	if growth == Growth::LeafWise && max_leaves == 0 {
		// By default, make leaf-wise comparable to a depth cap by allowing up to 2^depth leaves.
		// This is the theoretical maximum leaves in a full binary tree with depth `depth`.
		max_leaves = (1u64.checked_shl(depth).unwrap_or(u64::MAX).min(u32::MAX as u64)) as u32;
		max_leaves = max_leaves.max(2);
	}

	Args {
		task,
		rows,
		cols,
		num_classes,
		growth,
		depth,
		max_leaves,
		trees,
		seed,
		valid_fraction,
		ipc,
		parquet,
		libsvm,
		uci_machine,
		label0,
		xgb_gbtree_case,
		out,
		out_json,
	}
}

fn print_help_and_exit() -> ! {
	eprintln!(
		"quality_eval\n\n  --task regression|binary|multiclass\n\n  Data:\n    --synthetic <rows> <cols>\n    --ipc <path> (requires io-arrow)\n    --parquet <path> (requires io-parquet)\n    --libsvm <path> (label + index:value, 1-based indices)\n    --uci-machine <path> (UCI computer hardware machine.data CSV)\n    --label0 <path> (tab/space-separated: label first, then features)\n    --xgb-gbtree-case <name> (loads tests/test-cases/xgboost/gbtree/training/<name>.*)\n\n  Training:\n    --trees <n>\n    --growth depthwise|leafwise\n    --depth <d> (depthwise; also used to derive default --leaves)\n    --leaves <n> (leafwise; default: 2^depth)\n\n  Task:\n    --classes <k> (multiclass only)\n\n  Misc:\n    --seed <u64>\n    --valid <fraction>\n    --out <path>\n    --out-json <path>\n\nFeature-gated libs:\n  --features bench-xgboost enables XGBoost\n  --features bench-lightgbm enables LightGBM\n"
	);
	std::process::exit(0)
}

#[derive(Debug, Serialize)]
struct QualityRunJson {
	task: String,
	seed: u64,
	rows_train: usize,
	rows_valid: usize,
	cols: usize,
	trees: u32,
	growth: String,
	depth: u32,
	leaves: u32,
	booste_rs: LibraryResultJson,
	#[serde(skip_serializing_if = "Option::is_none")]
	xgboost: Option<LibraryResultJson>,
	#[serde(skip_serializing_if = "Option::is_none")]
	lightgbm: Option<LibraryResultJson>,
}

#[derive(Debug, Serialize)]
struct LibraryResultJson {
	#[serde(skip_serializing_if = "Option::is_none")]
	pred_kind: Option<String>,
	metrics: serde_json::Value,
}

fn extract_group(output: &booste_rs::inference::common::PredictionOutput, group: usize) -> Vec<f32> {
	let num_groups = output.num_groups();
	assert!(group < num_groups);
	if num_groups == 1 {
		return output.as_slice().to_vec();
	}

	(0..output.num_rows()).map(|r| output.row(r)[group]).collect()
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

#[derive(Debug, Deserialize)]
struct XgbDenseDataJson {
	num_rows: usize,
	num_features: usize,
	data: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct XgbLabelsJson {
	labels: Vec<f32>,
}

fn normalize_multiclass_labels(mut y: Vec<f32>, num_classes: usize) -> Vec<f32> {
	let mut min = f32::INFINITY;
	let mut max = f32::NEG_INFINITY;
	for &v in &y {
		min = min.min(v);
		max = max.max(v);
	}
	let k = num_classes as f32;
	// Common convention: labels are 1..K. Convert to 0..K-1.
	if (min - 1.0).abs() < 1e-6 && (max - k).abs() < 1e-6 {
		for v in &mut y {
			*v -= 1.0;
		}
	}

	for (i, &v) in y.iter().enumerate() {
		if v < 0.0 || v >= k {
			panic!("multiclass label out of range at idx={i}: got {v}, expected [0, {k})");
		}
	}
	// Metrics expect class indices as floats.
	y
}

fn load_xgb_dense_data(path: &Path) -> (Vec<f32>, usize, usize) {
	let content = fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
	let parsed: XgbDenseDataJson =
		serde_json::from_str(&content).unwrap_or_else(|e| panic!("failed to parse {}: {e}", path.display()));
	assert_eq!(parsed.data.len(), parsed.num_rows * parsed.num_features);
	(parsed.data, parsed.num_rows, parsed.num_features)
}

fn load_xgb_labels(path: &Path) -> Vec<f32> {
	let content = fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
	let parsed: XgbLabelsJson =
		serde_json::from_str(&content).unwrap_or_else(|e| panic!("failed to parse {}: {e}", path.display()));
	parsed.labels
}

fn load_xgb_gbtree_case(case: &str) -> (Vec<f32>, Vec<f32>, usize, Vec<f32>, Vec<f32>, usize, usize) {
	let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/test-cases/xgboost/gbtree/training");
	let train_data_path = base.join(format!("{case}.train_data.json"));
	let train_labels_path = base.join(format!("{case}.train_labels.json"));
	let test_data_path = base.join(format!("{case}.test_data.json"));
	let test_labels_path = base.join(format!("{case}.test_labels.json"));

	let (x_train, train_rows, train_cols) = load_xgb_dense_data(&train_data_path);
	let y_train = load_xgb_labels(&train_labels_path);
	assert_eq!(y_train.len(), train_rows);

	let (x_valid, valid_rows, valid_cols) = load_xgb_dense_data(&test_data_path);
	let y_valid = load_xgb_labels(&test_labels_path);
	assert_eq!(y_valid.len(), valid_rows);

	assert_eq!(train_cols, valid_cols, "train/test feature counts differ");
	(x_train, y_train, train_rows, x_valid, y_valid, valid_rows, train_cols)
}

fn load_or_generate(args: &Args) -> (Vec<f32>, Vec<f32>, usize, usize) {
	if let Some(path) = &args.libsvm {
		return load_libsvm_dense(path);
	}
	if let Some(path) = &args.uci_machine {
		return load_uci_machine_regression(path);
	}
	if let Some(path) = &args.label0 {
		return load_label0_dense(path);
	}

	#[cfg(feature = "io-arrow")]
	if let Some(path) = &args.ipc {
		return load_ipc_xy_row_major_f32(path).expect("failed to load ipc");
	}
	#[cfg(not(feature = "io-arrow"))]
	if args.ipc.is_some() {
		panic!("--ipc requires --features io-arrow");
	}

	#[cfg(feature = "io-parquet")]
	if let Some(path) = &args.parquet {
		return load_parquet_xy_row_major_f32(path).expect("failed to load parquet");
	}
	#[cfg(not(feature = "io-parquet"))]
	if args.parquet.is_some() {
		panic!("--parquet requires --features io-parquet");
	}

	let x = random_dense_f32(args.rows, args.cols, args.seed, -1.0, 1.0);
	let y = match args.task {
		Task::Regression => synthetic_regression_targets_linear(&x, args.rows, args.cols, args.seed ^ 0x0BAD_5EED, 0.05).0,
		Task::Binary => synthetic_binary_targets_from_linear_score(&x, args.rows, args.cols, args.seed ^ 0xB1A2_0001, 0.2),
		Task::Multiclass => synthetic_multiclass_targets_from_linear_scores(
			&x,
			args.rows,
			args.cols,
			args.num_classes,
			args.seed ^ 0x00C1_A550,
			0.1,
		),
	};
	(x, y, args.rows, args.cols)
}

fn load_libsvm_dense(path: &PathBuf) -> (Vec<f32>, Vec<f32>, usize, usize) {
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
				.unwrap_or_else(|| panic!("invalid feature token '{p}' at line {}", line_idx + 1));
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

fn load_label0_dense(path: &PathBuf) -> (Vec<f32>, Vec<f32>, usize, usize) {
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

fn load_uci_machine_regression(path: &PathBuf) -> (Vec<f32>, Vec<f32>, usize, usize) {
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

fn main() {
	let args = parse_args();

	let (x_train, y_train, rows_train, x_valid, y_valid, rows_valid, cols) = if let Some(case) = &args.xgb_gbtree_case {
		load_xgb_gbtree_case(case)
	} else {
		let (x_all, y_all, rows, cols) = load_or_generate(&args);
		let (train_idx, valid_idx) = split_indices(rows, args.valid_fraction, args.seed ^ 0x51EED);
		let x_train = select_rows_row_major(&x_all, rows, cols, &train_idx);
		let y_train = select_targets(&y_all, &train_idx);
		let x_valid = select_rows_row_major(&x_all, rows, cols, &valid_idx);
		let y_valid = select_targets(&y_all, &valid_idx);
		(x_train, y_train, train_idx.len(), x_valid, y_valid, valid_idx.len(), cols)
	};

	let y_train = match args.task {
		Task::Multiclass => normalize_multiclass_labels(y_train, args.num_classes),
		_ => y_train,
	};
	let y_valid = match args.task {
		Task::Multiclass => normalize_multiclass_labels(y_valid, args.num_classes),
		_ => y_valid,
	};

	let row_train: DenseMatrix<f32, RowMajor> = DenseMatrix::from_vec(x_train.clone(), rows_train, cols);
	let col_train: ColMatrix<f32> = row_train.to_layout();
	let binned_train = BinnedDatasetBuilder::from_matrix(&col_train, 256).build().unwrap();

	let row_valid: RowMatrix<f32> = RowMatrix::from_vec(x_valid.clone(), rows_valid, cols);

	let growth_strategy = match args.growth {
		Growth::DepthWise => GrowthStrategy::DepthWise { max_depth: args.depth },
		Growth::LeafWise => GrowthStrategy::LeafWise { max_leaves: args.max_leaves },
	};

	let params = GBDTParams {
		n_trees: args.trees,
		learning_rate: 0.1,
		growth_strategy,
		gain: GainParams { reg_lambda: 1.0, ..Default::default() },
		n_threads: 0,
		cache_size: 256,
		seed: args.seed,
		..Default::default()
	};

	let (boosters_metrics, boosters_metrics_json, boosters_pred_kind) = match args.task {
		Task::Regression => {
			let trainer = GBDTTrainer::new(SquaredLoss, params);
			let forest = trainer.train(&binned_train, &y_train, &[]).unwrap();
			let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);
			let pred = predictor.predict(&row_valid);
			let pred0: Vec<f32> = extract_group(&pred, 0);
			let n_rows = y_valid.len();
			let rmse = Rmse.compute(n_rows, 1, &pred0, &y_valid, &[]);
			let mae = Mae.compute(n_rows, 1, &pred0, &y_valid, &[]);
			(
				format!("rmse={rmse:.6} mae={mae:.6}"),
				serde_json::json!({"rmse": rmse, "mae": mae}),
				"raw".to_string(),
			)
		}
		Task::Binary => {
			let trainer = GBDTTrainer::new(LogisticLoss, params);
			let forest = trainer.train(&binned_train, &y_train, &[]).unwrap();
			let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);
			let raw = predictor.predict(&row_valid);
			let mut prob = extract_group(&raw, 0);
			sigmoid_inplace(&mut prob);
			let n_rows = y_valid.len();
			let ll = LogLoss.compute(n_rows, 1, &prob, &y_valid, &[]);
			let acc = Accuracy::default().compute(n_rows, 1, &prob, &y_valid, &[]);
			(
				format!(
					"logloss={ll:.6} acc={acc:.4}",
				),
				serde_json::json!({"logloss": ll, "acc": acc}),
				"prob".to_string(),
			)
		}
		Task::Multiclass => {
			let trainer = GBDTTrainer::new(SoftmaxLoss::new(args.num_classes), params);
			let forest = trainer.train(&binned_train, &y_train, &[]).unwrap();
			let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);
			let raw = predictor.predict(&row_valid);
			let n_rows = rows_valid;
			let mut prob_row_major = vec![0.0f32; n_rows * args.num_classes];
			assert_eq!(raw.num_groups(), args.num_classes);
			for r in 0..n_rows {
				let mut logits = raw.row(r).to_vec();
				softmax_inplace(&mut logits);
				prob_row_major[r * args.num_classes..(r + 1) * args.num_classes].copy_from_slice(&logits);
			}
			let ll = MulticlassLogLoss.compute(n_rows, args.num_classes, &prob_row_major, &y_valid, &[]);
			let acc = MulticlassAccuracy.compute(n_rows, args.num_classes, &prob_row_major, &y_valid, &[]);
			(
				format!(
					"mlogloss={ll:.6} acc={acc:.4}",
				),
				serde_json::json!({"mlogloss": ll, "acc": acc}),
				"prob".to_string(),
			)
		}
	};

	let display_leaves = match args.growth {
		Growth::DepthWise => (1u64.checked_shl(args.depth).unwrap_or(u64::MAX).min(u32::MAX as u64)) as u32,
		Growth::LeafWise => args.max_leaves,
	};

	println!("=== booste-rs ===");
	println!(
		"task={:?} rows_train={} rows_valid={} cols={} trees={} growth={:?} depth={} leaves={} pred_kind={}",
		args.task,
		rows_train,
		rows_valid,
		cols,
		args.trees,
		args.growth,
		args.depth,
		display_leaves,
		boosters_pred_kind
	);
	println!("metrics: {boosters_metrics}");

	let xgb_result: Option<LibraryResultJson> = {
		#[cfg(feature = "bench-xgboost")]
		{
		let (max_depth, max_leaves) = match args.growth {
			Growth::DepthWise => (args.depth, 0),
			// In XGBoost, leaf-wise growth is typically represented by `grow_policy=lossguide`
			// combined with `max_leaves`. To avoid imposing a depth cap that changes behavior,
			// use max_depth=0 (unlimited) and constrain only by max_leaves.
			Growth::LeafWise => (0, args.max_leaves),
		};
		let grow_policy = match args.growth {
			Growth::DepthWise => GrowPolicy::Depthwise,
			Growth::LeafWise => GrowPolicy::LossGuide,
		};

		let tree_params = TreeBoosterParametersBuilder::default()
			.eta(0.1)
			.max_depth(max_depth)
			.max_leaves(max_leaves)
			.grow_policy(grow_policy)
			.lambda(1.0)
			.alpha(0.0)
			.gamma(0.0)
			.min_child_weight(1.0)
			.tree_method(TreeMethod::Hist)
			.max_bin(256u32)
			.build()
			.unwrap();

		let objective = match args.task {
			Task::Regression => Objective::RegLinear,
			Task::Binary => Objective::BinaryLogistic,
			Task::Multiclass => Objective::MultiSoftprob(args.num_classes as u32),
		};
		let learning_params = LearningTaskParametersBuilder::default()
			.objective(objective)
			.build()
			.unwrap();

		let booster_params = BoosterParametersBuilder::default()
			.booster_type(BoosterType::Tree(tree_params))
			.learning_params(learning_params)
			.verbose(false)
			.threads(Some(1))
			.build()
			.unwrap();

		let mut dtrain = DMatrix::from_dense(&x_train, rows_train).unwrap();
		dtrain.set_labels(&y_train).unwrap();
		let mut dvalid = DMatrix::from_dense(&x_valid, rows_valid).unwrap();
		dvalid.set_labels(&y_valid).unwrap();

		let training_params = TrainingParametersBuilder::default()
			.dtrain(&dtrain)
			.boost_rounds(args.trees)
			.booster_params(booster_params)
			.evaluation_sets(None)
			.build()
			.unwrap();
		let model = XgbBooster::train(&training_params).unwrap();

		let pred = model.predict(&dvalid).unwrap();
		println!("=== xgboost ===");
		println!("task={:?} trees={} growth={:?} depth={} leaves={}", args.task, args.trees, args.growth, args.depth, display_leaves);
		let result = match args.task {
			Task::Regression => {
				let pred_f32: Vec<f32> = pred.into_iter().map(|x| x as f32).collect();
				let n_rows = y_valid.len();
				let rmse = Rmse.compute(n_rows, 1, &pred_f32, &y_valid, &[]);
				let mae = Mae.compute(n_rows, 1, &pred_f32, &y_valid, &[]);
				println!("metrics: rmse={rmse:.6} mae={mae:.6}");
				LibraryResultJson { pred_kind: Some("raw".to_string()), metrics: serde_json::json!({"rmse": rmse, "mae": mae}) }
			}
			Task::Binary => {
				let pred_f32: Vec<f32> = pred.into_iter().map(|x| x as f32).collect();
				let n_rows = y_valid.len();
				let ll = LogLoss.compute(n_rows, 1, &pred_f32, &y_valid, &[]);
				let acc = Accuracy::default().compute(n_rows, 1, &pred_f32, &y_valid, &[]);
				println!("metrics: logloss={ll:.6} acc={acc:.4}");
				LibraryResultJson { pred_kind: Some("prob".to_string()), metrics: serde_json::json!({"logloss": ll, "acc": acc}) }
			}
			Task::Multiclass => {
				let n_rows = rows_valid;
				let mut prob_row_major = vec![0.0f32; n_rows * args.num_classes];
				for (i, v) in pred.into_iter().enumerate() {
					prob_row_major[i] = v as f32;
				}
				let mut sum_min = f32::INFINITY;
				let mut sum_max = f32::NEG_INFINITY;
				for r in 0..n_rows {
					let start = r * args.num_classes;
					let sum: f32 = prob_row_major[start..start + args.num_classes].iter().sum();
					sum_min = sum_min.min(sum);
					sum_max = sum_max.max(sum);
				}
				println!("prob_sums: min={sum_min:.6} max={sum_max:.6}");
				let ll = MulticlassLogLoss.compute(n_rows, args.num_classes, &prob_row_major, &y_valid, &[]);
				let acc = MulticlassAccuracy.compute(n_rows, args.num_classes, &prob_row_major, &y_valid, &[]);
				println!("metrics: mlogloss={ll:.6} acc={acc:.4}");
				LibraryResultJson { pred_kind: Some("prob".to_string()), metrics: serde_json::json!({"mlogloss": ll, "acc": acc, "prob_sum_min": sum_min, "prob_sum_max": sum_max}) }
			}
		};
		Some(result)
		}
		#[cfg(not(feature = "bench-xgboost"))]
		{
			None
		}
	};

	let lgb_result: Option<LibraryResultJson> = {
		#[cfg(feature = "bench-lightgbm")]
		{
		let x_train_f64: Vec<f64> = x_train.iter().map(|&x| x as f64).collect();
		let x_valid_f64: Vec<f64> = x_valid.iter().map(|&x| x as f64).collect();
		let mut params = match args.task {
			Task::Regression => serde_json::json!({"objective":"regression","metric":"l2"}),
			Task::Binary => serde_json::json!({"objective":"binary","metric":"binary_logloss"}),
			Task::Multiclass => serde_json::json!({"objective":"multiclass","metric":"multi_logloss","num_class": args.num_classes}),
		};
		params["num_iterations"] = serde_json::Value::from(args.trees as i64);
		params["learning_rate"] = serde_json::Value::from(0.1f64);
		params["max_bin"] = serde_json::Value::from(256i64);
		match args.growth {
			Growth::DepthWise => {
				params["max_depth"] = serde_json::Value::from(args.depth as i64);
				// Avoid constraining depth-wise growth by `num_leaves`.
				let num_leaves = (1u64.checked_shl(args.depth).unwrap_or(u64::MAX)).max(2) as i64;
				params["num_leaves"] = serde_json::Value::from(num_leaves);
			}
			Growth::LeafWise => {
				params["max_depth"] = serde_json::Value::from(-1i64);
				params["num_leaves"] = serde_json::Value::from(args.max_leaves as i64);
			}
		}
		params["min_data_in_leaf"] = serde_json::Value::from(1i64);
		params["lambda_l2"] = serde_json::Value::from(1.0f64);
		params["feature_fraction"] = serde_json::Value::from(1.0f64);
		params["bagging_fraction"] = serde_json::Value::from(1.0f64);
		params["bagging_freq"] = serde_json::Value::from(0i64);
		params["verbosity"] = serde_json::Value::from(-1i64);
		params["num_threads"] = serde_json::Value::from(1i64);

		let ds = lightgbm3::Dataset::from_slice(&x_train_f64, &y_train, cols as i32, true).unwrap();
		let bst = lightgbm3::Booster::train(ds, &params).unwrap();

		let pred = bst.predict(&x_valid_f64, cols as i32, true).unwrap();
		println!("=== lightgbm ===");
		println!("task={:?} trees={} growth={:?} depth={} leaves={}", args.task, args.trees, args.growth, args.depth, display_leaves);
		let result = match args.task {
			Task::Regression => {
				let pred_f32: Vec<f32> = pred.into_iter().map(|x| x as f32).collect();
				let n_rows = y_valid.len();
				let rmse = Rmse.compute(n_rows, 1, &pred_f32, &y_valid, &[]);
				let mae = Mae.compute(n_rows, 1, &pred_f32, &y_valid, &[]);
				println!("metrics: rmse={rmse:.6} mae={mae:.6}");
				LibraryResultJson { pred_kind: Some("raw".to_string()), metrics: serde_json::json!({"rmse": rmse, "mae": mae}) }
			}
			Task::Binary => {
				let pred_f32: Vec<f32> = pred.into_iter().map(|x| x as f32).collect();
				let n_rows = y_valid.len();
				let ll = LogLoss.compute(n_rows, 1, &pred_f32, &y_valid, &[]);
				let acc = Accuracy::default().compute(n_rows, 1, &pred_f32, &y_valid, &[]);
				println!("metrics: logloss={ll:.6} acc={acc:.4}");
				LibraryResultJson { pred_kind: Some("prob".to_string()), metrics: serde_json::json!({"logloss": ll, "acc": acc}) }
			}
			Task::Multiclass => {
				let n_rows = rows_valid;
				let mut prob_row_major = vec![0.0f32; n_rows * args.num_classes];
				for (i, v) in pred.into_iter().enumerate() {
					prob_row_major[i] = v as f32;
				}
				let mut sum_min = f32::INFINITY;
				let mut sum_max = f32::NEG_INFINITY;
				for r in 0..n_rows {
					let start = r * args.num_classes;
					let sum: f32 = prob_row_major[start..start + args.num_classes].iter().sum();
					sum_min = sum_min.min(sum);
					sum_max = sum_max.max(sum);
				}
				println!("prob_sums: min={sum_min:.6} max={sum_max:.6}");
				let ll = MulticlassLogLoss.compute(n_rows, args.num_classes, &prob_row_major, &y_valid, &[]);
				let acc = MulticlassAccuracy.compute(n_rows, args.num_classes, &prob_row_major, &y_valid, &[]);
				println!("metrics: mlogloss={ll:.6} acc={acc:.4}");
				LibraryResultJson { pred_kind: Some("prob".to_string()), metrics: serde_json::json!({"mlogloss": ll, "acc": acc, "prob_sum_min": sum_min, "prob_sum_max": sum_max}) }
			}
		};
		Some(result)
		}
		#[cfg(not(feature = "bench-lightgbm"))]
		{
			None
		}
	};

	if let Some(out) = &args.out {
		let content = format!(
			"task={:?}\nrows_train={} rows_valid={} cols={} trees={} growth={:?} depth={} leaves={}\nbooste_rs: {}\n",
			args.task,
			rows_train,
			rows_valid,
			cols,
			args.trees,
			args.growth,
			args.depth,
			display_leaves,
			boosters_metrics
		);
		fs::write(out, content).expect("write out");
		println!("wrote {}", out.display());
	}

	if let Some(out_json) = &args.out_json {
		let growth = match args.growth {
			Growth::DepthWise => "depthwise",
			Growth::LeafWise => "leafwise",
		};
		let task = match args.task {
			Task::Regression => "regression",
			Task::Binary => "binary",
			Task::Multiclass => "multiclass",
		};
		let run = QualityRunJson {
			task: task.to_string(),
			seed: args.seed,
			rows_train,
			rows_valid,
			cols,
			trees: args.trees,
			growth: growth.to_string(),
			depth: args.depth,
			leaves: display_leaves,
			booste_rs: LibraryResultJson {
				pred_kind: Some(boosters_pred_kind),
				metrics: boosters_metrics_json,
			},
			xgboost: xgb_result,
			lightgbm: lgb_result,
		};
		let content = serde_json::to_string_pretty(&run).expect("serialize json");
		fs::write(out_json, content).expect("write out-json");
		println!("wrote {}", out_json.display());
	}
}
