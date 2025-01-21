# Quality Benchmark Report

**Seeds**: 5 ([42, 1379, 2716, 4053, 5390])

## REGRESSION

### RMSE (lower is better)

| Dataset | booste-rs | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| regression_small | 2.368046 ± 0.257735 | **2.360284 ± 0.260442** | 2.363582 ± 0.264341 |
| regression_medium | 3.361773 ± 0.131533 | **3.361749 ± 0.130092** | 3.362367 ± 0.132094 |
| regression_linear_small | **2.378271 ± 0.259013** | - | - |
| regression_linear_medium | **3.366265 ± 0.131868** | - | - |

### MAE (lower is better)

| Dataset | booste-rs | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| regression_small | 1.885451 ± 0.201668 | **1.880783 ± 0.205457** | 1.883935 ± 0.210653 |
| regression_medium | 2.684438 ± 0.107092 | **2.683367 ± 0.105953** | 2.684967 ± 0.107115 |
| regression_linear_small | **1.893258 ± 0.203795** | - | - |
| regression_linear_medium | **2.688252 ± 0.107353** | - | - |

## BINARY

### LogLoss (lower is better)

| Dataset | booste-rs | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| binary_small | **0.716001 ± 0.003919** | 0.719919 ± 0.005672 | 0.717703 ± 0.002702 |
| binary_medium | 0.700174 ± 0.000901 | 0.699942 ± 0.001526 | **0.699780 ± 0.001851** |

### Accuracy (higher is better)

| Dataset | booste-rs | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| binary_small | 0.5010 ± 0.0087 | 0.4985 ± 0.0136 | **0.5026 ± 0.0103** |
| binary_medium | 0.5026 ± 0.0045 | 0.5038 ± 0.0082 | **0.5058 ± 0.0082** |

## MULTICLASS

### Multi-class LogLoss (lower is better)

| Dataset | booste-rs | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| multiclass_small | 1.691774 ± 0.005754 | **1.650830 ± 0.003558** | 1.684946 ± 0.005226 |
| multiclass_medium | 1.640108 ± 0.005423 | **1.620561 ± 0.001347** | 1.631651 ± 0.002025 |

### Accuracy (higher is better)

| Dataset | booste-rs | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| multiclass_small | 0.2071 ± 0.0047 | **0.2094 ± 0.0047** | 0.2071 ± 0.0070 |
| multiclass_medium | 0.2024 ± 0.0067 | **0.2085 ± 0.0071** | 0.2069 ± 0.0045 |

## Benchmark Configuration

| Dataset | Data Source | Trees | Depth | Classes | Linear |
|---------|-------------|-------|-------|---------|--------|
| regression_small | Synthetic 10000x50 | 100 | 6 | - | - |
| regression_medium | Synthetic 50000x100 | 100 | 6 | - | - |
| regression_linear_small | Synthetic 10000x50 | 100 | 6 | - | ✓ |
| regression_linear_medium | Synthetic 50000x100 | 100 | 6 | - | ✓ |
| binary_small | Synthetic 10000x50 | 100 | 6 | - | - |
| binary_medium | Synthetic 50000x100 | 100 | 6 | - | - |
| multiclass_small | Synthetic 10000x50 | 100 | 6 | 5 | - |
| multiclass_medium | Synthetic 50000x100 | 100 | 6 | 5 | - |

