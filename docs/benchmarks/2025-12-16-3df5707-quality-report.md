# Quality Benchmark Report

**Seeds**: 5 ([42, 1379, 2716, 4053, 5390])

## REGRESSION

### RMSE (lower is better)

| Dataset | boosters | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| regression_small | **1.015405 ± 0.069788** | 1.027186 ± 0.068480 | 1.032413 ± 0.069189 |
| regression_medium | **1.815355 ± 0.073498** | 1.819204 ± 0.077371 | 1.823884 ± 0.072398 |
| california_housing | 0.504099 ± 0.006097 | 0.478709 ± 0.010811 | **0.478660 ± 0.010602** |

### MAE (lower is better)

| Dataset | boosters | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| regression_small | **0.807530 ± 0.050502** | 0.816810 ± 0.052308 | 0.822858 ± 0.052458 |
| regression_medium | **1.448366 ± 0.058498** | 1.451396 ± 0.063510 | 1.455081 ± 0.058819 |
| california_housing | 0.340131 ± 0.006714 | **0.315868 ± 0.004037** | 0.316560 ± 0.004083 |

## BINARY

### LogLoss (lower is better)

| Dataset | boosters | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| binary_small | 0.335367 ± 0.013207 | **0.335070 ± 0.014210** | 0.338852 ± 0.017024 |
| binary_medium | 0.419154 ± 0.011042 | **0.418536 ± 0.009643** | 0.419307 ± 0.010694 |
| adult | 0.282153 ± 0.006308 | **0.278670 ± 0.006186** | 0.278884 ± 0.006493 |

### Accuracy (higher is better)

| Dataset | boosters | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| binary_small | 0.8695 ± 0.0103 | **0.8736 ± 0.0082** | 0.8675 ± 0.0127 |
| binary_medium | 0.8465 ± 0.0048 | **0.8466 ± 0.0032** | 0.8466 ± 0.0052 |
| adult | 0.8707 ± 0.0049 | **0.8730 ± 0.0042** | 0.8725 ± 0.0047 |

## MULTICLASS

### Multi-class LogLoss (lower is better)

| Dataset | boosters | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| multiclass_small | **0.627704 ± 0.012191** | 2.319503 ± 0.018802 | 2.727644 ± 0.018809 |
| multiclass_medium | **0.765417 ± 0.010978** | 1.939270 ± 0.010944 | 2.148563 ± 0.014719 |
| covertype | **0.420593 ± 0.006009** | 5.202451 ± 0.020138 | 6.865731 ± 0.038856 |

### Accuracy (higher is better)

| Dataset | boosters | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| multiclass_small | **0.7585 ± 0.0056** | 0.1953 ± 0.0080 | 0.2006 ± 0.0073 |
| multiclass_medium | **0.7463 ± 0.0066** | 0.2009 ± 0.0028 | 0.2002 ± 0.0012 |
| covertype | **0.8250 ± 0.0036** | 0.1438 ± 0.0038 | 0.1436 ± 0.0038 |

## Benchmark Configuration

| Dataset | Data Source | Trees | Depth | Classes |
|---------|-------------|-------|-------|--------|
| regression_small | Synthetic 10000x50 | 100 | 6 | - |
| regression_medium | Synthetic 50000x100 | 100 | 6 | - |
| binary_small | Synthetic 10000x50 | 100 | 6 | - |
| binary_medium | Synthetic 50000x100 | 100 | 6 | - |
| multiclass_small | Synthetic 10000x50 | 100 | 6 | 5 |
| multiclass_medium | Synthetic 50000x100 | 100 | 6 | 5 |
| california_housing | california_housing | 100 | 6 | - |
| adult | adult | 100 | 6 | - |
| covertype | covertype (subsampled to 50000) | 100 | 6 | 7 |

