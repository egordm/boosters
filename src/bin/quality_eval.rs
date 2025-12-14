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
use std::path::PathBuf;

use booste_rs::data::{binned::BinnedDatasetBuilder, ColMatrix, DenseMatrix, RowMajor, RowMatrix};
use booste_rs::inference::common::{sigmoid_inplace, softmax_inplace};
use booste_rs::inference::gbdt::{Predictor, UnrolledTraversal6};
use booste_rs::testing::data::{
	random_dense_f32, split_indices, synthetic_binary_targets_from_linear_score,
	synthetic_multiclass_targets_from_linear_scores, synthetic_regression_targets_linear,
};
use booste_rs::training::{GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, LogisticLoss, SoftmaxLoss, SquaredLoss};

use booste_rs::training::{Accuracy, LogLoss, Mae, Metric, MulticlassAccuracy, MulticlassLogLoss, Rmse};

#[cfg(feature = "io-arrow")]
use booste_rs::data::io::arrow::load_ipc_xy_row_major_f32;
#[cfg(feature = "io-parquet")]
use booste_rs::data::io::parquet::load_parquet_xy_row_major_f32;

#[cfg(feature = "bench-xgboost")]
use xgb::parameters::tree::{TreeBoosterParametersBuilder, TreeMethod};
#[cfg(feature = "bench-xgboost")]
use xgb::parameters::{learning::LearningTaskParametersBuilder, learning::Objective, BoosterParametersBuilder, TrainingParametersBuilder};
#[cfg(feature = "bench-xgboost")]
use xgb::parameters::BoosterType;
#[cfg(feature = "bench-xgboost")]
use xgb::{Booster as XgbBooster, DMatrix};

#[cfg(feature = "bench-lightgbm")]
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Task {
	Regression,
	Binary,
	Multiclass,
}

#[derive(Debug)]
struct Args {
	task: Task,
	rows: usize,
	cols: usize,
	num_classes: usize,
	depth: u32,
	trees: u32,
	seed: u64,
	valid_fraction: f32,
	ipc: Option<PathBuf>,
	parquet: Option<PathBuf>,
	out: Option<PathBuf>,
}

fn parse_args() -> Args {
	let mut task = Task::Regression;
	let mut rows = 50_000usize;
	let mut cols = 100usize;
	let mut num_classes = 3usize;
	let mut depth = 6u32;
	let mut trees = 200u32;
	let mut seed = 42u64;
	let mut valid_fraction = 0.2f32;
	let mut ipc: Option<PathBuf> = None;
	let mut parquet: Option<PathBuf> = None;
	let mut out: Option<PathBuf> = None;

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
			"--synthetic" => {
				rows = it.next().expect("--synthetic rows").parse().unwrap();
				cols = it.next().expect("--synthetic cols").parse().unwrap();
			}
			"--classes" => num_classes = it.next().expect("--classes value").parse().unwrap(),
			"--depth" => depth = it.next().expect("--depth value").parse().unwrap(),
			"--trees" => trees = it.next().expect("--trees value").parse().unwrap(),
			"--seed" => seed = it.next().expect("--seed value").parse().unwrap(),
			"--valid" => valid_fraction = it.next().expect("--valid value").parse().unwrap(),
			"--ipc" => ipc = Some(PathBuf::from(it.next().expect("--ipc path"))),
			"--parquet" => parquet = Some(PathBuf::from(it.next().expect("--parquet path"))),
			"--out" => out = Some(PathBuf::from(it.next().expect("--out path"))),
			"--help" => {
				print_help_and_exit();
			}
			other => panic!("unknown arg: {other}"),
		}
	}

	Args {
		task,
		rows,
		cols,
		num_classes,
		depth,
		trees,
		seed,
		valid_fraction,
		ipc,
		parquet,
		out,
	}
}

fn print_help_and_exit() -> ! {
	eprintln!(
		"quality_eval\n\n  --task regression|binary|multiclass\n  --synthetic <rows> <cols>\n  --ipc <path> (requires io-arrow)\n  --parquet <path> (requires io-parquet)\n  --trees <n>\n  --depth <d>\n  --classes <k> (multiclass only)\n  --seed <u64>\n  --valid <fraction>\n  --out <path>\n\nFeature-gated libs:\n  --features bench-xgboost enables XGBoost\n  --features bench-lightgbm enables LightGBM\n"
	);
	std::process::exit(0)
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

fn load_or_generate(args: &Args) -> (Vec<f32>, Vec<f32>, usize, usize) {
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

fn main() {
	let args = parse_args();

	let (x_all, y_all, rows, cols) = load_or_generate(&args);
	let (train_idx, valid_idx) = split_indices(rows, args.valid_fraction, args.seed ^ 0x51EED);
	let x_train = select_rows_row_major(&x_all, rows, cols, &train_idx);
	let y_train = select_targets(&y_all, &train_idx);
	let x_valid = select_rows_row_major(&x_all, rows, cols, &valid_idx);
	let y_valid = select_targets(&y_all, &valid_idx);

	let row_train: DenseMatrix<f32, RowMajor> = DenseMatrix::from_vec(x_train.clone(), train_idx.len(), cols);
	let col_train: ColMatrix<f32> = row_train.to_layout();
	let binned_train = BinnedDatasetBuilder::from_matrix(&col_train, 256).build().unwrap();

	let row_valid: RowMatrix<f32> = RowMatrix::from_vec(x_valid.clone(), valid_idx.len(), cols);

	let params = GBDTParams {
		n_trees: args.trees,
		learning_rate: 0.1,
		growth_strategy: GrowthStrategy::DepthWise { max_depth: args.depth },
		gain: GainParams { reg_lambda: 1.0, ..Default::default() },
		n_threads: 0,
		cache_size: 256,
		seed: args.seed,
		..Default::default()
	};

	let (boosters_metrics, boosters_pred_kind) = match args.task {
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
				"prob".to_string(),
			)
		}
		Task::Multiclass => {
			let trainer = GBDTTrainer::new(SoftmaxLoss::new(args.num_classes), params);
			let forest = trainer.train(&binned_train, &y_train, &[]).unwrap();
			let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);
			let raw = predictor.predict(&row_valid);
			let n_rows = valid_idx.len();
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
				"prob".to_string(),
			)
		}
	};

	println!("=== booste-rs ===");
	println!("task={:?} rows_train={} rows_valid={} cols={} trees={} depth={} pred_kind={}", args.task, train_idx.len(), valid_idx.len(), cols, args.trees, args.depth, boosters_pred_kind);
	println!("metrics: {boosters_metrics}");

	// Optional: XGBoost
	#[cfg(feature = "bench-xgboost")]
	{
		let tree_params = TreeBoosterParametersBuilder::default()
			.eta(0.1)
			.max_depth(args.depth)
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

		let mut dtrain = DMatrix::from_dense(&x_train, train_idx.len()).unwrap();
		dtrain.set_labels(&y_train).unwrap();
		let mut dvalid = DMatrix::from_dense(&x_valid, valid_idx.len()).unwrap();
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
		match args.task {
			Task::Regression => {
				let pred_f32: Vec<f32> = pred.into_iter().map(|x| x as f32).collect();
				let n_rows = y_valid.len();
				let rmse = Rmse.compute(n_rows, 1, &pred_f32, &y_valid, &[]);
				let mae = Mae.compute(n_rows, 1, &pred_f32, &y_valid, &[]);
				println!("metrics: rmse={rmse:.6} mae={mae:.6}");
			}
			Task::Binary => {
				let pred_f32: Vec<f32> = pred.into_iter().map(|x| x as f32).collect();
				let n_rows = y_valid.len();
				let ll = LogLoss.compute(n_rows, 1, &pred_f32, &y_valid, &[]);
				let acc = Accuracy::default().compute(n_rows, 1, &pred_f32, &y_valid, &[]);
				println!("metrics: logloss={ll:.6} acc={acc:.4}");
			}
			Task::Multiclass => {
				let n_rows = valid_idx.len();
				let mut prob_row_major = vec![0.0f32; n_rows * args.num_classes];
				for (i, v) in pred.into_iter().enumerate() {
					prob_row_major[i] = v as f32;
				}
				let ll = MulticlassLogLoss.compute(n_rows, args.num_classes, &prob_row_major, &y_valid, &[]);
				let acc = MulticlassAccuracy.compute(n_rows, args.num_classes, &prob_row_major, &y_valid, &[]);
				println!("metrics: mlogloss={ll:.6} acc={acc:.4}");
			}
		}
	}

	// Optional: LightGBM
	#[cfg(feature = "bench-lightgbm")]
	{
		let x_train_f64: Vec<f64> = x_train.iter().map(|&x| x as f64).collect();
		let x_valid_f64: Vec<f64> = x_valid.iter().map(|&x| x as f64).collect();
		let mut params = match args.task {
			Task::Regression => json!({"objective":"regression","metric":"l2"}),
			Task::Binary => json!({"objective":"binary","metric":"binary_logloss"}),
			Task::Multiclass => json!({"objective":"multiclass","metric":"multi_logloss","num_class": args.num_classes}),
		};
		params["num_iterations"] = serde_json::Value::from(args.trees as i64);
		params["learning_rate"] = serde_json::Value::from(0.1f64);
		params["max_depth"] = serde_json::Value::from(args.depth as i64);
		params["num_leaves"] = serde_json::Value::from(31i64);
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
		match args.task {
			Task::Regression => {
				let pred_f32: Vec<f32> = pred.into_iter().map(|x| x as f32).collect();
				let n_rows = y_valid.len();
				let rmse = Rmse.compute(n_rows, 1, &pred_f32, &y_valid, &[]);
				let mae = Mae.compute(n_rows, 1, &pred_f32, &y_valid, &[]);
				println!("metrics: rmse={rmse:.6} mae={mae:.6}");
			}
			Task::Binary => {
				let pred_f32: Vec<f32> = pred.into_iter().map(|x| x as f32).collect();
				let n_rows = y_valid.len();
				let ll = LogLoss.compute(n_rows, 1, &pred_f32, &y_valid, &[]);
				let acc = Accuracy::default().compute(n_rows, 1, &pred_f32, &y_valid, &[]);
				println!("metrics: logloss={ll:.6} acc={acc:.4}");
			}
			Task::Multiclass => {
				let n_rows = valid_idx.len();
				let mut prob_row_major = vec![0.0f32; n_rows * args.num_classes];
				for (i, v) in pred.into_iter().enumerate() {
					prob_row_major[i] = v as f32;
				}
				let ll = MulticlassLogLoss.compute(n_rows, args.num_classes, &prob_row_major, &y_valid, &[]);
				let acc = MulticlassAccuracy.compute(n_rows, args.num_classes, &prob_row_major, &y_valid, &[]);
				println!("metrics: mlogloss={ll:.6} acc={acc:.4}");
			}
		}
	}

	if let Some(out) = &args.out {
		let content = format!(
			"task={:?}\nrows_train={} rows_valid={} cols={} trees={} depth={}\nbooste_rs: {}\n",
			args.task,
			train_idx.len(),
			valid_idx.len(),
			cols,
			args.trees,
			args.depth,
			boosters_metrics
		);
		fs::write(out, content).expect("write out");
		println!("wrote {}", out.display());
	}
}
