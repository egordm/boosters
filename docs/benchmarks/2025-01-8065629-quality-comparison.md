# Quality Benchmark Report

**Date**: 2025-01-19  
**Commit**: 8065629  
**Machine**: Apple M1 Pro (10 cores), 32GB RAM  
**Seeds**: 3 ([42, 1379, 2716])

## Executive Summary

This report compares model quality between **booste-rs**, **XGBoost**, and **LightGBM**
on both synthetic and real-world datasets. All libraries use equivalent hyperparameters
(100 trees, max_depth=6, learning_rate=0.1).

### Key Findings

| Task | Winner | Notes |
|------|--------|-------|
| Regression (synthetic) | **booste-rs** | ~1% better RMSE |
| Regression (real-world) | XGBoost/LightGBM | ~6% better on California Housing |
| Binary (synthetic) | Tie | XGBoost marginally better (<0.1%) |
| Binary (real-world) | XGBoost/LightGBM | ~0.2% better accuracy on Adult |
| Multiclass | **booste-rs** | 2-4% better accuracy across all datasets |
| Linear GBDT | **booste-rs only** | XGBoost doesn't support, LightGBM crate crashes |

### Real-World Dataset Performance

| Dataset | Samples | Features | Classes | Best Library |
|---------|---------|----------|---------|--------------|
| California Housing | 20,640 | 8 | Regression | XGBoost |
| Adult | 48,842 | 105 | Binary | LightGBM |
| Covertype | 50,000* | 54 | 7 | **booste-rs** |

*Covertype subsampled for reasonable benchmark time.

---

## REGRESSION

### RMSE (lower is better)

| Dataset | booste-rs | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| regression_small | **0.999636 ± 0.092380** | 1.015481 ± 0.092585 | 1.022252 ± 0.094190 |
| regression_medium | **1.823381 ± 0.080050** | 1.827738 ± 0.079322 | 1.830870 ± 0.078924 |
| regression_linear_small | **0.763377 ± 0.065951** | - | - |
| regression_linear_medium | **1.432749 ± 0.069333** | - | - |
| california_housing | 0.503698 ± 0.003829 | **0.474507 ± 0.007832** | 0.475351 ± 0.009275 |
| california_housing_linear | **0.512032 ± 0.015323** | - | - |

### MAE (lower is better)

| Dataset | booste-rs | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| regression_small | **0.793993 ± 0.066031** | 0.805637 ± 0.070615 | 0.813891 ± 0.071763 |
| regression_medium | **1.454927 ± 0.063135** | 1.458201 ± 0.063888 | 1.462277 ± 0.062619 |
| regression_linear_small | **0.609777 ± 0.050557** | - | - |
| regression_linear_medium | **1.143787 ± 0.056021** | - | - |
| california_housing | 0.341069 ± 0.006675 | **0.314370 ± 0.001649** | 0.315473 ± 0.002041 |
| california_housing_linear | **0.342311 ± 0.005905** | - | - |

## BINARY

### LogLoss (lower is better)

| Dataset | booste-rs | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| binary_small | 0.328589 ± 0.005422 | **0.328395 ± 0.005466** | 0.330699 ± 0.007964 |
| binary_medium | 0.413588 ± 0.007895 | **0.412948 ± 0.006405** | 0.414083 ± 0.007670 |
| adult | 0.277658 ± 0.001806 | 0.274326 ± 0.002170 | **0.274306 ± 0.002213** |

### Binary Accuracy (higher is better)

| Dataset | booste-rs | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| binary_small | 0.8747 ± 0.0051 | **0.8770 ± 0.0079** | 0.8722 ± 0.0078 |
| binary_medium | **0.8488 ± 0.0041** | 0.8488 ± 0.0018 | 0.8480 ± 0.0031 |
| adult | 0.8743 ± 0.0008 | **0.8760 ± 0.0010** | 0.8759 ± 0.0005 |

## MULTICLASS

### Multi-class LogLoss (lower is better)

| Dataset | booste-rs | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| multiclass_small | **0.629409 ± 0.016919** | 0.768474 ± 0.020721 | 0.665694 ± 0.023927 |
| multiclass_medium | **0.771716 ± 0.003243** | 0.959966 ± 0.005767 | 0.831726 ± 0.003892 |
| covertype | **0.421125 ± 0.006400** | 0.473723 ± 0.001356 | 0.429375 ± 0.005677 |

### Accuracy (higher is better)

| Dataset | booste-rs | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| multiclass_small | **0.7575 ± 0.0077** | 0.7375 ± 0.0026 | 0.7523 ± 0.0122 |
| multiclass_medium | **0.7433 ± 0.0048** | 0.7033 ± 0.0074 | 0.7342 ± 0.0072 |
| covertype | **0.8245 ± 0.0031** | 0.8001 ± 0.0011 | 0.8203 ± 0.0053 |

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
| california_housing | california_housing | 100 | 6 | - | - |
| california_housing_linear | california_housing | 100 | 6 | - | ✓ |
| adult | adult | 100 | 6 | - | - |
| covertype | covertype (subsampled to 50000) | 100 | 6 | 7 | - |

