# Post-Refactor Performance Comparison

**Date**: 2025-12-20  
**Commit**: c6cc704 (post-refactor)  
**Baseline**: f89b89d (pre-refactor)

---

## Summary

**No significant regression detected.** All benchmarks within 5% tolerance.

---

## Prediction Core Benchmarks

| Benchmark | Pre-Refactor | Post-Refactor | Change |
|-----------|--------------|---------------|--------|
| batch_size/medium/1 | 9.05 µs | 8.83 µs | **-2.4%** ✅ |
| batch_size/medium/10 | 13.30 µs | 13.55 µs | +1.9% ✅ |
| batch_size/medium/100 | 87.67 µs | 89.46 µs | +2.0% ✅ |
| batch_size/medium/1000 | 868.1 µs | 885.49 µs | +2.0% ✅ |
| batch_size/medium/10000 | 8.68 ms | 8.96 ms | +3.2% ✅ |
| model_size/small/1000 | 92.37 µs | 92.85 µs | +0.5% ✅ |
| model_size/medium/1000 | 868.4 µs | 899.30 µs | +3.6% ✅ |

**Verdict**: All within ±5% threshold. Changes are within measurement noise.

---

## Analysis

The slight regression (2-4%) in some benchmarks is likely due to:

1. **Measurement noise** - Criterion reports "change within noise threshold" for most
2. **Additional indirection** - `Metric::is_enabled()` check adds a branch
3. **System variability** - Different system load between measurements

The API refactor did NOT introduce significant performance degradation.

---

## Correctness Verification

```bash
cargo test --test compat
```

All XGBoost compatibility tests pass with unchanged expected values.

---

## Conclusion

**Story 9.1 complete.** No action needed - performance is acceptable.
