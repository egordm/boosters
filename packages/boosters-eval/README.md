# boosters-eval

Evaluation benchmarks for comparing gradient boosting libraries.

## Overview

This package provides an extensible framework for evaluating and comparing gradient boosting
libraries (XGBoost, LightGBM, and boosters) across various datasets and configurations.

## Installation

```bash
# Basic installation
uv pip install -e .

# With all supported libraries
uv pip install -e ".[all]"
```

## Usage

```bash
# Run quality benchmark with default settings
boosters-eval quality --seeds 5

# Quick mode for development
boosters-eval quality --quick --seeds 3

# Output to markdown file
boosters-eval quality --seeds 5 --output report.md

# Run specific benchmark modes
boosters-eval quality --mode synthetic  # Synthetic datasets only
boosters-eval quality --mode real       # Real-world datasets only
```

## Extensibility

The benchmark framework is designed to be extensible:

- **Datasets**: Add new datasets by implementing the `Dataset` protocol
- **Runners**: Add new library runners by implementing the `Runner` protocol  
- **Metrics**: Add new metrics by implementing the `Metric` protocol
- **Benchmarks**: Create new benchmark suites by composing datasets, runners, and metrics

## Package Structure

```text
src/boosters_eval/
├── benchmarks/     # Benchmark configurations and suites
├── datasets/       # Dataset loaders (synthetic, parquet, etc.)
├── metrics/        # Evaluation metrics (RMSE, LogLoss, Accuracy, etc.)
├── runners/        # Library-specific training/prediction runners
├── reports/        # Report generation (markdown, JSON, etc.)
└── cli.py          # Typer CLI application
```
