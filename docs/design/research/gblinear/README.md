# GBLinear Research

This folder contains research on XGBoost's linear booster (GBLinear).

## Contents

- [overview.md](overview.md) — What is GBLinear? When to use it?
- [inference.md](inference.md) — How prediction works
- [training.md](training.md) — Coordinate descent training algorithms
- [parameters.md](parameters.md) — Configuration options and their effects
- [optimizations.md](optimizations.md) — Performance optimizations (threading, GPU, SIMD)

## Key Concepts

**GBLinear** is a generalized linear model with elastic net regularization (L1 + L2).
Unlike GBTree which builds an ensemble of decision trees, GBLinear learns feature
weights directly — essentially a regularized linear regression or logistic regression.

### ELI5

Imagine you're guessing a house price. GBTree looks at rules like "if bedrooms > 3
AND near park, then +$50k". GBLinear just says "each bedroom adds $20k, each bathroom
adds $15k" and multiplies. Much simpler, but less powerful for complex patterns.

## Source Files (XGBoost C++)

| File | Purpose |
|------|---------|
| `src/gbm/gblinear.cc` | Main booster implementation |
| `src/gbm/gblinear_model.h` | Weight matrix structure |
| `src/linear/param.h` | Training parameters |
| `src/linear/coordinate_common.h` | Update algorithms |
| `src/linear/updater_shotgun.cc` | Parallel updater |
| `src/linear/updater_coordinate.cc` | Sequential updater |
