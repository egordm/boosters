# Pre-Refactor Baseline

**Date**: 2025-12-20  
**Commit**: f89b89d  
**Purpose**: Capture performance and correctness baselines before the data/API layering redesign (now RFC-0021)

---

## Performance Baselines

### Prediction Core (`cargo bench --bench prediction_core`)

| Benchmark | Time | Throughput |
| --------- | ---- | ---------- |
| batch_size/medium/1 | 9.05 µs | 110.5 Kelem/s |
| batch_size/medium/10 | 13.30 µs | 751.8 Kelem/s |
| batch_size/medium/100 | 87.67 µs | 1.14 Melem/s |
| batch_size/medium/1000 | 868.1 µs | 1.15 Melem/s |
| batch_size/medium/10000 | 8.68 ms | 1.15 Melem/s |
| model_size/small/1000 | 92.37 µs | 10.83 Melem/s |
| model_size/medium/1000 | 868.4 µs | 1.15 Melem/s |

### Training GBDT (`cargo bench --bench training_gbdt`)

| Benchmark | Time | Throughput |
| --------- | ---- | ---------- |
| quantize/to_binned/max_bins=256/10000x50 | 32.24 ms | 15.51 Melem/s |
| quantize/to_binned/max_bins=256/50000x100 | 328.4 ms | 15.23 Melem/s |
| quantize/to_binned/max_bins=256/100000x20 | 132.9 ms | 15.05 Melem/s |
| train_regression/train/small | 261.0 ms | 1.92 Melem/s |
| train_regression/train/medium | 1.11 s | 4.49 Melem/s |

---

## Correctness Baselines

### Selected XGBoost Compatibility Test Cases

The following 3 test cases are selected for regression testing:

1. **gbtree_regression** - Basic regression model
2. **gbtree_binary_logistic** - Binary classification with logistic loss
3. **gbtree_multiclass** - Multiclass classification (3 classes)

### Test Case Locations

- `tests/test-cases/xgboost/gbtree/inference/gbtree_regression.expected.json`
- `tests/test-cases/xgboost/gbtree/inference/gbtree_binary_logistic.expected.json`
- `tests/test-cases/xgboost/gbtree/inference/gbtree_multiclass.expected.json`

### Verification Commands

```bash
# Run selected XGBoost compat tests
cargo test --test compat -- predict_regression predict_binary_logistic predict_multiclass

# Run all compatibility tests
cargo test --test compat

# Run all tests
cargo test
```

### Test Results

All 3 selected tests pass with tolerance `< 1e-5`:

```text
test xgboost::predict_regression ... ok
test xgboost::predict_binary_logistic ... ok
test xgboost::predict_multiclass ... ok
```

---

## Post-Refactor Comparison Instructions

After completing the API refactor, run:

1. **Performance check**:

   ```bash
   cargo bench --bench prediction_core
   cargo bench --bench training_gbdt
   ```

   Compare against baselines above. Accept if regression < 5%.

2. **Correctness check**:

   ```bash
   cargo test --test compat
   ```

   All tests must pass with unchanged expected values.

3. **Full test suite**:

   ```bash
   cargo test
   ```

   All tests must pass.

---

## Notes

- This baseline was captured before any API refactoring changes
- The XGBoost `.expected.json` files serve as ground truth for prediction outputs
- Any post-refactor prediction changes > 1e-5 tolerance indicate a regression
