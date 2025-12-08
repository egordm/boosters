# 2025-12-02: Column-Major Predictions Layout

## Goal

Test whether changing prediction buffer layout from row-major to column-major
impacts loss computation performance. Motivation: consistency with column-major
GradientBuffer and potential cache benefits.

## Summary

Column-major predictions are **faster** for loss computation, not slower.
Softmax sees 1.5-3x improvement, quantile loss sees 4-5x improvement.
The consistent column-major layout provides perfect sequential access for
both reads (predictions) and writes (gradients).

## Environment

| Property | Value |
|----------|-------|
| CPU | Apple M1 Pro (10-core) |
| RAM | 32GB |
| OS | macOS |
| Rust | 1.82.0 |
| Commit | ab5b37f |

## Results

### Softmax Loss (Multiclass)

| Size | Row-Major | Col-Major | Speedup |
|------|-----------|-----------|---------|
| 1000×3 | 16.7 µs | 8.1 µs | 2.1x |
| 1000×10 | 53.4 µs | 28.9 µs | 1.8x |
| 1000×100 | 435 µs | 285 µs | 1.5x |
| 10000×3 | 177 µs | 90.9 µs | 1.9x |
| 10000×10 | 499 µs | 288 µs | 1.7x |
| 10000×100 | 5.23 ms | 3.02 ms | 1.7x |
| 100000×3 | 1.66 ms | 895 µs | 1.9x |
| 100000×10 | 4.99 ms | 3.07 ms | 1.6x |
| 100000×100 | 92.8 ms | 29.5 ms | **3.1x** |

### Quantile Loss (Multi-Quantile Regression)

| Size | Row-Major | Col-Major | Speedup |
|------|-----------|-----------|---------|
| 1000×3 | 2.25 µs | 440 ns | **5.1x** |
| 10000×3 | 22.9 µs | 5.02 µs | **4.6x** |
| 100000×3 | 222 µs | 58 µs | **3.8x** |

### Logistic Loss (Single Output)

Single output - layout doesn't matter (~414 Melem/s).

## Analysis

Row-major predictions require strided reads when iterating by output
(accessing `preds[i * n_outputs + k]` for varying `i`). Column-major
allows sequential reads (`preds[k * n_samples + i]`) matching the
gradient buffer's column-major layout.

Quantile loss benefits most because it's pure streaming: for each quantile,
read contiguous predictions, compare with labels, write contiguous gradients.

## Conclusions

Switching predictions to column-major:
- Improves loss computation performance significantly
- Provides consistency with gradient buffer layout
- Simplifies mental model for multi-output training

Decision: Implement column-major predictions for both GBTree and GBLinear.
