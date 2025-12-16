//! Comprehensive quality benchmark runner.
//!
//! This script runs quality_eval across multiple configurations and seeds,
//! aggregating results with confidence intervals.
//!
//! Usage:
//!   cargo run --bin quality_benchmark --release --features "bench-xgboost,bench-lightgbm" -- [options]
//!
//! Options:
//!   --seeds <n>         Number of seeds to run (default: 5)
//!   --out <path>        Output markdown file (default: stdout)
//!   --quick             Quick mode: fewer rows, fewer trees

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use serde::{Deserialize, Serialize};

// =============================================================================
// Configuration
// =============================================================================

#[derive(Debug, Clone)]
struct BenchmarkConfig {
	name: String,
	task: &'static str,
	rows: usize,
	cols: usize,
	trees: u32,
	depth: u32,
	classes: Option<usize>,
}

fn default_configs(quick: bool) -> Vec<BenchmarkConfig> {
	let (small_rows, medium_rows) = if quick { (2_000, 10_000) } else { (10_000, 50_000) };
	let trees = if quick { 50 } else { 100 };

	vec![
		// Regression benchmarks
		BenchmarkConfig {
			name: "regression_small".to_string(),
			task: "regression",
			rows: small_rows,
			cols: 50,
			trees,
			depth: 6,
			classes: None,
		},
		BenchmarkConfig {
			name: "regression_medium".to_string(),
			task: "regression",
			rows: medium_rows,
			cols: 100,
			trees,
			depth: 6,
			classes: None,
		},
		// Binary classification benchmarks
		BenchmarkConfig {
			name: "binary_small".to_string(),
			task: "binary",
			rows: small_rows,
			cols: 50,
			trees,
			depth: 6,
			classes: None,
		},
		BenchmarkConfig {
			name: "binary_medium".to_string(),
			task: "binary",
			rows: medium_rows,
			cols: 100,
			trees,
			depth: 6,
			classes: None,
		},
		// Multiclass classification benchmarks
		BenchmarkConfig {
			name: "multiclass_small".to_string(),
			task: "multiclass",
			rows: small_rows,
			cols: 50,
			trees,
			depth: 6,
			classes: Some(5),
		},
		BenchmarkConfig {
			name: "multiclass_medium".to_string(),
			task: "multiclass",
			rows: medium_rows,
			cols: 100,
			trees,
			depth: 6,
			classes: Some(5),
		},
	]
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
	min: f64,
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
// Execution
// =============================================================================

fn run_quality_eval(config: &BenchmarkConfig, seed: u64) -> Option<RunResult> {
	let manifest_dir = env!("CARGO_MANIFEST_DIR");
	let tmp_json = format!("/tmp/quality_eval_{}.json", std::process::id());

	let mut cmd = Command::new("cargo");
	cmd.current_dir(manifest_dir);
	cmd.args([
		"run",
		"--bin",
		"quality_eval",
		"--release",
		"--features",
		"bench-xgboost,bench-lightgbm",
		"--",
		"--task",
		config.task,
		"--synthetic",
		&config.rows.to_string(),
		&config.cols.to_string(),
		"--trees",
		&config.trees.to_string(),
		"--depth",
		&config.depth.to_string(),
		"--seed",
		&seed.to_string(),
		"--out-json",
		&tmp_json,
	]);

	if let Some(classes) = config.classes {
		cmd.args(["--classes", &classes.to_string()]);
	}

	let output = cmd.output().ok()?;
	if !output.status.success() {
		eprintln!(
			"quality_eval failed for {} seed {}: {}",
			config.name,
			seed,
			String::from_utf8_lossy(&output.stderr)
		);
		return None;
	}

	let json_content = fs::read_to_string(&tmp_json).ok()?;
	let _ = fs::remove_file(&tmp_json);

	serde_json::from_str(&json_content).ok()
}

fn run_benchmark(config: &BenchmarkConfig, seeds: &[u64]) -> BenchmarkResult {
	println!("Running benchmark: {} ({} seeds)", config.name, seeds.len());

	let mut boosters_runs: Vec<LibraryMetrics> = Vec::new();
	let mut xgb_runs: Vec<LibraryMetrics> = Vec::new();
	let mut lgb_runs: Vec<LibraryMetrics> = Vec::new();

	for (i, &seed) in seeds.iter().enumerate() {
		print!("  Seed {}/{}: {} ... ", i + 1, seeds.len(), seed);
		std::io::stdout().flush().unwrap();

		if let Some(result) = run_quality_eval(config, seed) {
			boosters_runs.push(result.booste_rs);
			if let Some(xgb) = result.xgboost {
				xgb_runs.push(xgb);
			}
			if let Some(lgb) = result.lightgbm {
				lgb_runs.push(lgb);
			}
			println!("OK");
		} else {
			println!("FAILED");
		}
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
	let valids: Vec<(usize, f64)> = vals
		.iter()
		.enumerate()
		.filter_map(|(i, opt)| opt.map(|s| (i, s.mean)))
		.collect();

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
		by_task.entry(r.config.task).or_default().push(r);
	}

	for (task, task_results) in &by_task {
		out.push_str(&format!("## {}\n\n", task.to_uppercase()));

		match *task {
			"regression" => {
				// RMSE table
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

				// MAE table
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
				// LogLoss table
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

				// Accuracy table
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
				// Multi-class LogLoss table
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

				// Accuracy table
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
	out.push_str("| Dataset | Rows | Features | Trees | Depth | Classes |\n");
	out.push_str("|---------|------|----------|-------|-------|--------|\n");
	for r in results {
		out.push_str(&format!(
			"| {} | {} | {} | {} | {} | {} |\n",
			r.config.name,
			r.config.rows,
			r.config.cols,
			r.config.trees,
			r.config.depth,
			r.config.classes.map(|c| c.to_string()).unwrap_or("-".to_string()),
		));
	}
	out.push_str("\n");

	out
}

// =============================================================================
// Main
// =============================================================================

fn parse_args() -> (usize, Option<PathBuf>, bool) {
	let mut num_seeds = 5usize;
	let mut out: Option<PathBuf> = None;
	let mut quick = false;

	let mut it = std::env::args().skip(1);
	while let Some(arg) = it.next() {
		match arg.as_str() {
			"--seeds" => num_seeds = it.next().expect("--seeds value").parse().unwrap(),
			"--out" => out = Some(PathBuf::from(it.next().expect("--out path"))),
			"--quick" => quick = true,
			"--help" => {
				eprintln!("quality_benchmark\n\n  --seeds <n>   Number of seeds (default: 5)\n  --out <path>  Output markdown file\n  --quick       Quick mode (fewer rows/trees)");
				std::process::exit(0);
			}
			other => panic!("unknown arg: {other}"),
		}
	}

	(num_seeds, out, quick)
}

fn main() {
	let (num_seeds, out_path, quick) = parse_args();

	// Generate seeds
	let seeds: Vec<u64> = (0..num_seeds).map(|i| 42 + i as u64 * 1337).collect();

	let configs = default_configs(quick);

	println!("=== Quality Benchmark ===");
	println!("Running {} configurations with {} seeds each", configs.len(), seeds.len());
	println!();

	let mut results: Vec<BenchmarkResult> = Vec::new();
	for config in &configs {
		results.push(run_benchmark(config, &seeds));
	}

	let report = generate_report(&results, &seeds);

	if let Some(path) = out_path {
		fs::write(&path, &report).expect("failed to write report");
		println!("\nReport written to: {}", path.display());
	} else {
		println!("\n{}", report);
	}
}
