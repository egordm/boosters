# boosters-eval

A simple, extensible framework for benchmarking gradient boosting libraries.

## Quick Start

```bash
# Install
pip install -e packages/boosters-eval

# Run quick benchmark
boosters-eval quick

# Compare specific datasets
boosters-eval compare -d california -d breast_cancer -l boosters -l xgboost
```

## Installation

```bash
# From the repository root
pip install -e packages/boosters-eval

# Dependencies (automatically installed)
# - boosters
# - xgboost>=2
# - lightgbm>=4
# - scikit-learn>=1.5
# - pydantic>=2
# - typer>=0.12
# - rich>=13
```

## CLI Commands

### Quick Benchmark

Run a quick benchmark suite (3 seeds, 2 datasets, 50 trees):

```bash
boosters-eval quick
boosters-eval quick -o results.md
```

### Full Benchmark

Run the full benchmark suite (5 seeds, all datasets, 100 trees):

```bash
boosters-eval full
boosters-eval full -o results.md

# Run with a specific booster type
boosters-eval full --booster gbdt          # Gradient boosted trees (default)
boosters-eval full --booster gblinear      # Linear booster
boosters-eval full --booster linear_trees  # Trees with linear leaves (LightGBM only)
```

### Compare Libraries

Compare specific libraries on selected datasets:

```bash
# Compare all libraries on california dataset
boosters-eval compare -d california

# Compare boosters vs xgboost with specific settings
boosters-eval compare -d california -d breast_cancer -l boosters -l xgboost --trees 100 --seeds 5

# Use a specific booster type
boosters-eval compare -d california --booster gblinear
```

### Baseline Management

Record and check baselines for CI regression detection:

```bash
# Record baseline
boosters-eval baseline record -o baseline.json -s quick

# Check against baseline
boosters-eval baseline check baseline.json -s quick --tolerance 0.02
```

### Generate Reports

Generate markdown reports with machine fingerprinting:

```bash
boosters-eval report -s quick -o docs/benchmarks/report.md
boosters-eval report -s full --title "Release 0.1.0 Benchmark"
boosters-eval report --dry-run  # Preview without saving
```

### List Resources

```bash
boosters-eval list-datasets    # Show available datasets
boosters-eval list-libraries   # Show available libraries
boosters-eval list-tasks       # Show supported ML tasks
```

## Python API

### Basic Usage

```python
from boosters_eval import compare, QUICK_SUITE, run_suite

# Quick comparison
results = compare(["california"], seeds=[42])
print(results.to_markdown())

# Run predefined suite
results = run_suite(QUICK_SUITE)
print(results.summary())

# Export results
results.to_json("results.json")
results.to_csv("results.csv")
```

### Custom Suites

```python
from boosters_eval import SuiteConfig, run_suite, BoosterType

suite = SuiteConfig(
    name="custom",
    description="My custom benchmark",
    datasets=["california", "breast_cancer"],
    n_estimators=100,
    seeds=[42, 123, 456],
    libraries=["boosters", "xgboost", "lightgbm"],
    booster_type=BoosterType.GBDT,
)

results = run_suite(suite)
```

### Ablation Studies

Run ablation studies to compare different hyperparameter settings:

```python
from boosters_eval import QUICK_SUITE, create_ablation_suite, run_suite

# Compare different tree depths
depth_variants = {
    "depth_4": {"max_depth": 4},
    "depth_6": {"max_depth": 6},
    "depth_8": {"max_depth": 8},
}
depth_suites = create_ablation_suite("depth_study", QUICK_SUITE, depth_variants)
for suite in depth_suites:
    results = run_suite(suite)
    print(f"{suite.name}:")
    print(results.to_markdown())

# Compare learning rates
lr_variants = {
    "lr_0.01": {"learning_rate": 0.01},
    "lr_0.1": {"learning_rate": 0.1},
    "lr_0.3": {"learning_rate": 0.3},
}
lr_suites = create_ablation_suite("lr_study", QUICK_SUITE, lr_variants)

# Compare number of trees
tree_variants = {
    "trees_50": {"n_estimators": 50},
    "trees_100": {"n_estimators": 100},
    "trees_200": {"n_estimators": 200},
}
tree_suites = create_ablation_suite("tree_study", QUICK_SUITE, tree_variants)
```

Available parameters for ablation:
- `n_estimators`: Number of boosting rounds
- `max_depth`: Maximum tree depth (default: 6)
- `learning_rate`: Step size shrinkage (default: 0.1)
- `booster_type`: BoosterType.GBDT, GBLINEAR, or LINEAR_TREES
- `datasets`: List of dataset names to include
- `libraries`: List of libraries to compare
```

### Baseline Regression Testing

```python
from boosters_eval import (
    record_baseline,
    load_baseline,
    check_baseline,
    run_suite,
    QUICK_SUITE,
)
from pathlib import Path

# Record baseline
results = run_suite(QUICK_SUITE)
baseline = record_baseline(results, output_path=Path("baseline.json"))

# Later: check for regressions
current_results = run_suite(QUICK_SUITE)
baseline = load_baseline(Path("baseline.json"))
report = check_baseline(current_results, baseline, tolerance=0.02)

if report.has_regressions:
    for reg in report.regressions:
        print(f"Regression: {reg['config']} {reg['metric']}")
```

## Available Datasets

| Dataset | Task | Size |
|---------|------|------|
| california | Regression | 20,640 samples |
| breast_cancer | Binary Classification | 569 samples |
| iris | Multiclass Classification | 150 samples |
| synthetic_reg_* | Synthetic Regression | Various sizes |
| synthetic_bin_* | Synthetic Binary | Various sizes |
| synthetic_multi_* | Synthetic Multiclass | Various sizes |

## Supported Libraries

| Library | Booster Types | Notes |
|---------|--------------|-------|
| boosters | gbdt, gblinear | Native Rust implementation |
| xgboost | gbdt, gblinear | Industry standard |
| lightgbm | gbdt, linear_trees | Leaf-wise growth |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Regression detected / Error |
| 2 | File not found |

## Troubleshooting

### Library Not Found

If a library is not showing as available:

```bash
# Check which libraries are installed
boosters-eval list-libraries

# Install missing libraries
pip install xgboost lightgbm
```

### Memory Issues

For large datasets, you may need to subsample:

```python
from boosters_eval import DatasetConfig, Task

# Custom dataset with subsampling
config = DatasetConfig(
    name="large_data",
    task=Task.REGRESSION,
    loader=my_loader,
    subsample=10000,  # Limit to 10k samples
)
```

### Baseline Version Mismatch

If you see a schema version error when loading a baseline:

```
Baseline schema version X is newer than supported version Y
```

Update boosters-eval to the latest version or re-record the baseline.

## Development

```bash
# Run tests
cd packages/boosters-eval
uv run pytest

# Run with coverage
uv run pytest --cov

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/
```

## License

MIT License - see repository root for details.
