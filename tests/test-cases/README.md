# Test Cases

This directory contains test data for validating booste-rs against reference implementations.

## Structure

```
test-cases/
├── xgboost/
│   ├── gbtree/
│   │   ├── inference/   # Model loading & prediction tests
│   │   └── training/    # Training validation tests
│   ├── gblinear/
│   │   ├── inference/
│   │   └── training/
│   └── dart/
│       └── inference/
└── benchmark/           # Benchmark models (various sizes)
```

## File Naming Convention

### Inference Test Cases

Each test case consists of three files:
- `{name}.model.json` - XGBoost model in JSON format
- `{name}.input.json` - Test input features
- `{name}.expected.json` - Expected predictions (raw margin + transformed)

### Training Test Cases

Each training test case consists of:
- `{name}.train_data.json` - Training features
- `{name}.train_labels.json` - Training labels  
- `{name}.config.json` - Training hyperparameters
- `{name}.model.json` - XGBoost trained model
- `{name}.train_predictions.json` - Predictions on training set
- `{name}.test_data.json` - Held-out test features (optional)
- `{name}.test_labels.json` - Held-out test labels (optional)
- `{name}.test_predictions.json` - Predictions on test set (optional)
- `{name}.metrics.json` - Test metrics (RMSE, accuracy, etc.)

## Regenerating Test Cases

Test cases are generated using Python scripts in `tools/data_generation/`.

Requirements:
- [uv](https://github.com/astral-sh/uv) package manager
- Python 3.11+

To regenerate all test cases:

```bash
cd tools/data_generation
uv run python scripts/generate_test_cases.py
```

To regenerate specific categories:

```bash
# XGBoost inference tests
uv run python -c "from scripts.generate_xgboost import generate_all_gbtree; generate_all_gbtree()"

# GBTree training tests
uv run python -c "from scripts.generate_xgboost import generate_all_gbtree_training; generate_all_gbtree_training()"

# GBLinear training tests  
uv run python -c "from scripts.generate_xgboost import generate_all_training; generate_all_training()"

# Benchmark models
uv run python -c "from scripts.generate_xgboost import generate_all_benchmarks; generate_all_benchmarks()"
```

## Test Case Categories

### GBTree Inference

| Name | Features | Trees | Description |
|------|----------|-------|-------------|
| gbtree_regression | 5 | 5 | Basic regression |
| gbtree_binary_logistic | 4 | 5 | Binary classification |
| gbtree_multiclass | 4 | 4 | 3-class softmax |
| gbtree_*_missing | varies | varies | Tests with NaN values |
| gbtree_categorical* | varies | 5 | Categorical feature tests |
| gbtree_deep_trees | 8 | 5 | max_depth=6 |
| gbtree_many_trees | 5 | 50 | Tests accumulation precision |
| gbtree_wide_features | 100 | 5 | High feature count |

### GBTree Training

| Name | Rows | Features | Trees | Description |
|------|------|----------|-------|-------------|
| regression_simple | 80 | 5 | 20 | Basic depth-wise |
| regression_deep | 160 | 10 | 50 | max_depth=6 |
| regression_regularized | 160 | 20 | 50 | L1+L2 regularization |
| binary_classification | 160 | 10 | 30 | Logistic objective |
| multiclass | 240 | 10 | 30 | 3-class softmax |
| leaf_wise | 160 | 10 | 30 | grow_policy=lossguide |
| large | 800 | 20 | 100 | Performance test |

### GBLinear Training

| Name | Rows | Features | Description |
|------|------|----------|-------------|
| regression_simple | 5 | 1 | y = 2x + 1 |
| regression_multifeature | 8 | 2 | y = x0 + 2*x1 + 3 |
| regression_l2 | 80 | 5 | L2 regularization |
| regression_elastic_net | 80 | 10 | L1 + L2 |
| binary_classification | 120 | 4 | Logistic |
| multiclass_classification | 150 | 4 | 3-class softmax |
| quantile_* | 120 | 5 | Quantile regression (α=0.1, 0.5, 0.9) |
