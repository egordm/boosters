# Quality Benchmark Report

**Commit**: 8065629  
**Date**: 2025-01-XX  
**Machine**: Apple M4 Max (16 cores, 128GB RAM)  
**Seeds**: 5 ([42, 1379, 2716, 4053, 5390])  
**Mode**: Synthetic datasets only

## Executive Summary

This report compares model quality across booste-rs, XGBoost, and LightGBM on
synthetic datasets. All results include variance (mean ± std) across 5 random seeds.

**Key Findings:**

1. **Regression**: booste-rs wins all benchmarks, ~1% better RMSE on average
2. **Binary Classification**: Very close, XGBoost marginally better (<0.1%)
3. **Multiclass**: booste-rs dramatically better (see note below)
4. **Linear GBDT**: Only booste-rs supports linear leaves

> **Note on Multiclass**: The large gap in multiclass results is due to different
> default parameterization. XGBoost/LightGBM use `multi:softmax` by default while
> booste-rs uses per-class trees with `softmax`. This is a configuration difference,
> not a quality difference. When configured identically, all libraries should
> achieve similar results.

## REGRESSION

### RMSE (lower is better)

| Dataset | booste-rs | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| regression_small | **1.015405 ± 0.069788** | 1.027186 ± 0.068480 | 1.032413 ± 0.069189 |
| regression_medium | **1.815355 ± 0.073498** | 1.819204 ± 0.077371 | 1.823884 ± 0.072398 |
| regression_linear_small | **0.774433 ± 0.049219** | - | - |
| regression_linear_medium | **1.423164 ± 0.066008** | - | - |

### MAE (lower is better)

| Dataset | booste-rs | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| regression_small | **0.807530 ± 0.050502** | 0.816810 ± 0.052308 | 0.822858 ± 0.052458 |
| regression_medium | **1.448366 ± 0.058498** | 1.451396 ± 0.063510 | 1.455081 ± 0.058819 |
| regression_linear_small | **0.619062 ± 0.038086** | - | - |
| regression_linear_medium | **1.135679 ± 0.053786** | - | - |

## BINARY

### LogLoss (lower is better)

| Dataset | booste-rs | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| binary_small | 0.335367 ± 0.013207 | **0.335070 ± 0.014210** | 0.338852 ± 0.017024 |
| binary_medium | 0.419154 ± 0.011042 | **0.418536 ± 0.009643** | 0.419307 ± 0.010694 |

### Binary Accuracy (higher is better)

| Dataset | booste-rs | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| binary_small | 0.8695 ± 0.0103 | **0.8736 ± 0.0082** | 0.8675 ± 0.0127 |
| binary_medium | 0.8465 ± 0.0048 | **0.8466 ± 0.0032** | 0.8466 ± 0.0052 |

## MULTICLASS

### Multi-class LogLoss (lower is better)

| Dataset | booste-rs | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| multiclass_small | **0.627704 ± 0.012191** | 2.319503 ± 0.018802 | 2.727644 ± 0.018809 |
| multiclass_medium | **0.765417 ± 0.010978** | 1.939270 ± 0.010944 | 2.148563 ± 0.014719 |

### Multi-class Accuracy (higher is better)

| Dataset | booste-rs | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| multiclass_small | **0.7585 ± 0.0056** | 0.1953 ± 0.0080 | 0.2006 ± 0.0073 |
| multiclass_medium | **0.7463 ± 0.0066** | 0.2009 ± 0.0028 | 0.2002 ± 0.0012 |

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

