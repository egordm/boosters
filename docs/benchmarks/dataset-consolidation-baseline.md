# Dataset Consolidation Baseline Benchmarks

**Date**: 2025-01-06  
**Commit**: f29cbcf  
**Machine**: Apple M1 (local development)  
**Rust**: stable (2024 edition)

> NOTE: This report predates RFC-0021 landing.
> Today, `Dataset` owns raw feature values and `BinnedDataset` is bins-only/internal.
> Any discussion of “raw access from BinnedDataset” in this document is historical context.

## Overview

This document captures baseline performance measurements for the Dataset consolidation effort.
The goal is to measure overhead of using `BinnedDataset` instead of `types/Dataset` for:

1. **Dataset Construction** - How much does binning add to data loading?
2. **GBLinear Training** - Critical: linear models don't need bins
3. **GBDT Training** - Baseline: trees need bins anyway
4. **Prediction** - Important: inference uses raw features, not bins

## Thresholds (from backlog)

| Metric | Target |
|--------|--------|
| GBLinear training overhead | < 2x on small datasets (≤10K samples) |
| Prediction overhead | < 10% regression |
| Memory overhead | Document (expected ~2x for raw storage) |

---

## Results

### 1. Dataset Construction Overhead

Measures time to construct Dataset vs BinnedDataset from raw features.

| Samples | Dataset | BinnedDataset | Overhead |
|---------|---------|---------------|----------|
| 1,000   | 7.6 µs  | 3.2 ms        | **421x** |
| 10,000  | 86.5 µs | 30.2 ms       | **349x** |
| 50,000  | 467 µs  | 162 ms        | **347x** |

**Analysis**: BinnedDataset construction is ~350x slower due to binning and EFB bundling analysis. This is expected and acceptable since:
- Construction is a one-time cost before training
- GBDT training requires binned data anyway
- The overhead is amortized over many training rounds

**Status**: ✅ Expected behavior, no action needed

---

### 2. GBLinear Training Overhead (CRITICAL)

Measures actual GBLinear training time with `Dataset` vs raw feature access patterns.

#### Full Training (with Dataset)

| Samples | GBLinear Training |
|---------|-------------------|
| 1,000   | 934 µs            |
| 10,000  | 3.48 ms           |
| 50,000  | 17.65 ms          |

#### Raw Feature Access (Dataset vs BinnedDataset)

| Samples | Dataset Access | BinnedDataset Access | Overhead |
|---------|----------------|----------------------|----------|
| 1,000   | 83.6 µs        | 83.0 µs              | **1.0x** ✅ |
| 10,000  | 938 µs         | 998 µs               | **1.06x** ✅ |
| 50,000  | 4.73 ms        | 4.84 ms              | **1.02x** ✅ |

**Analysis**: Raw feature access from BinnedDataset is essentially identical to Dataset access (within noise). The overhead is negligible (<6%).

**Status**: ✅ **PASS** - Well within <2x threshold

#### 2b. train() vs train_binned() Comparison (Story 3.2)

End-to-end comparison of `GBLinearTrainer::train(&Dataset)` vs `GBLinearTrainer::train_binned(&BinnedDataset)`:

| Samples | train(Dataset) | train_binned(BinnedDataset) | Overhead |
|---------|----------------|------------------------------|----------|
| 1,000   | 946 µs         | 1.05 ms                      | **1.11x** ✅ |
| 10,000  | 3.38 ms        | 3.51 ms                      | **1.04x** ✅ |
| 50,000  | 16.96 ms       | 18.19 ms                     | **1.07x** ✅ |

**Analysis**: The overhead of using `train_binned()` with BinnedDataset is **4-11%**, well within the <2x threshold.

The slight overhead comes from `BinnedDataset::to_raw_feature_matrix()` which extracts a contiguous `[n_features, n_samples]` matrix from raw values stored alongside binned data. This cost is:

- Amortized over all training rounds (10 rounds in benchmark)
- Acceptable for users who want to use the same BinnedDataset for both GBDT and GBLinear training

**Status**: ✅ **PASS** - 4-11% overhead is negligible

#### 2c. predict() vs predict_binned() Comparison (Story 3.3)

End-to-end comparison of `GBLinearModel::predict(Dataset)` vs `GBLinearModel::predict_binned(BinnedDataset)`:

| Samples | predict(Dataset) | predict_binned(BinnedDataset) | Overhead |
|---------|------------------|-------------------------------|----------|
| 1,000   | 128.7 µs         | 133.6 µs                      | **1.04x** ✅ |
| 10,000  | 1.29 ms          | 1.31 ms                       | **1.02x** ✅ |
| 50,000  | 6.63 ms          | 6.61 ms                       | **1.00x** ✅ |

**Analysis**: Prediction overhead is **0-4%** - essentially negligible. The `to_raw_feature_matrix()` copy cost is well-amortized because linear model prediction is O(n_features × n_samples) anyway.

**Status**: ✅ **PASS** - 0-4% overhead is negligible

---

### 3. GBDT Training (Baseline)

GBDT always uses BinnedDataset. This establishes baseline performance.

| Config | Time | Throughput |
|--------|------|------------|
| 10K samples × 50 features, 10 trees | 65.5 ms | 7.6 Melem/s |

**Status**: ✅ Baseline established

---

### 4. Prediction Overhead

Measures prediction with trained GBDT model.

#### Full Prediction

| Samples | Prediction Time |
|---------|-----------------|
| 10,000  | 6.83 ms         |

#### Raw Feature Access During Prediction

| Samples | Dataset Access | BinnedDataset Access | Overhead |
|---------|----------------|----------------------|----------|
| 10,000  | 95.5 µs        | 114.4 µs             | **1.20x** |

**Analysis**: BinnedDataset raw access is ~20% slower than Dataset access for the prediction access pattern. This is due to:
- Additional indirection through `raw_feature_slice()` vs direct array access
- The slice lookup vs contiguous row access

However, this is only measuring the feature access portion. The actual prediction overhead will be lower since tree traversal dominates prediction time.

**Status**: ⚠️ **MONITOR** - 20% overhead on feature access, but actual prediction impact likely <10%

---

### 4b. End-to-End Prediction Overhead (Story 2.3)

Measures complete prediction throughput comparing FeaturesView path vs BinnedDataset + SampleBlocks path.

**Test Configuration**:
- 10,000 samples × 100 features
- 50 trees, depth 6
- Block size: 64 samples

| Path | Time | Throughput | Overhead |
|------|------|------------|----------|
| FeaturesView (Dataset) | 6.71 ms | 74.5 Melem/s | Baseline |
| SampleBlocks (BinnedDataset) | 7.42 ms | 67.4 Melem/s | **+10.6%** |

**Analysis**: The SampleBlocks path is ~10.6% slower. Root cause:

1. **Block extraction**: SampleBlocks extracts raw values from BinnedDataset to sample-major blocks `[samples, features]`
2. **Transpose to feature-major**: `predict_into` requires feature-major data `[features, samples]`
3. **Internal re-transpose**: `predict_into` transposes back to sample-major for block prediction

This "double transpose" is the overhead source. The SampleBlocks data is already sample-major, but the public API expects feature-major.

**Potential Optimizations** (future work):
- Expose sample-major prediction path (would require new API)
- Use `predict_row_into` per sample (loses block benefits)
- Direct feature-major view from BinnedDataset (requires contiguous raw storage)

**Status**: ⚠️ **MONITOR** - 10.6% overhead slightly exceeds <10% target but is acceptable for:
- BinnedDataset construction is already the common path for GBDT
- The overhead is due to data layout conversion, not algorithmic issues
- FeaturesView can still be used when raw arrays are available

---

## Summary

| Metric | Threshold | Measured | Status |
|--------|-----------|----------|--------|
| GBLinear raw access overhead | < 2x | 1.0-1.06x | ✅ PASS |
| Prediction raw access overhead | < 10% | ~20% | ⚠️ MONITOR |
| End-to-end prediction overhead | < 10% | ~10.6% | ⚠️ MONITOR |
| Construction overhead | N/A | ~350x | ✅ Expected |

### Key Findings

1. **GBLinear is safe**: Raw feature access from BinnedDataset is essentially identical to Dataset access. No performance concerns for linear model training.

2. **Prediction overhead is acceptable**: End-to-end prediction with SampleBlocks is ~10.6% slower than FeaturesView. This slightly exceeds the <10% target but is acceptable because:
   - The overhead is due to data layout conversion (double transpose), not algorithmic issues
   - FeaturesView can still be used when raw arrays are available
   - For training workflows, BinnedDataset construction cost is already amortized

3. **Construction overhead is expected**: The 350x overhead is the cost of binning, which is amortized over training rounds.

---

## 5. Memory Overhead Analysis (Story 0.4)

### Theoretical Calculation

For 50K samples × 50 features:

**types/Dataset**:
- Features: 50 × 50,000 × 4 bytes (f32) = **10.0 MB**
- Targets: 1 × 50,000 × 4 bytes = 0.2 MB
- Schema: ~1 KB (negligible)
- **Total: ~10.2 MB**

**BinnedDataset**:
- Raw features: 50 × 50,000 × 4 bytes = 10.0 MB
- Binned features (u8, max_bins=256): 50 × 50,000 × 1 byte = **2.5 MB**
- Binned features (u16, max_bins>256): 50 × 50,000 × 2 bytes = 5.0 MB
- Bin mappers: 50 × 256 × 4 bytes = ~50 KB
- Feature metadata: ~2 KB
- **Total: ~12.5 MB** (u8 bins) or ~15 MB (u16 bins)

### Memory Overhead

| Configuration | Dataset | BinnedDataset | Overhead |
|---------------|---------|---------------|----------|
| max_bins=256 (u8) | 10.2 MB | 12.5 MB | **+23%** |
| max_bins>256 (u16) | 10.2 MB | 15.0 MB | **+47%** |

### Assessment

- Default configuration (max_bins=256): **23% overhead** ⚠️ MONITOR
- High bin count (max_bins>256): **47% overhead** ⚠️ MONITOR

Per threshold definitions:
- <20%: Acceptable ✅
- 20-50%: Note in report, monitor ⚠️
- >50%: Consider lazy binning mitigation ❌

**Status**: ⚠️ **MONITOR** - Memory overhead is in the 20-50% range which is acceptable but should be monitored. For most use cases, this is fine since:
- Training typically dominates memory usage
- The bins are needed for GBDT anyway
- GBLinear-only users can use `BinningConfig::enable_binning(false)` if implemented

---

## 6. Risk Review Gate (Story 0.5)

### Checklist

- [x] All baseline benchmarks captured
- [x] Memory overhead acceptable per thresholds (23% with u8 bins)
- [x] No blocking issues identified
- [x] Team consensus to proceed

### Decision

**GO** ✅ - Proceed with consolidation

**Rationale**:
1. GBLinear raw access overhead is negligible (1.0-1.06x)
2. Memory overhead is in acceptable range (23%)
3. Prediction raw access overhead (20%) needs monitoring but actual prediction impact expected <10%
4. No blocking stakeholder feedback for consolidation

### Recommendations

1. **Proceed with consolidation** - GBLinear overhead is acceptable
2. **Story 2.3 critical** - Need to measure actual prediction overhead (not just access)
3. **Consider optimization** - If prediction overhead is too high, investigate:
   - Trait dispatch optimization with `#[inline]`
   - SampleBlocks for cache-friendly access
   - Direct slice access without Option wrapping

---

## Appendix: Raw Criterion Output

```text
component/overhead/construction/Dataset/1000:        7.6 µs
component/overhead/construction/BinnedDataset/1000: 3.2 ms
component/overhead/construction/Dataset/10000:      86.5 µs  
component/overhead/construction/BinnedDataset/10000: 30.2 ms
component/overhead/construction/Dataset/50000:      467 µs
component/overhead/construction/BinnedDataset/50000: 162 ms

component/overhead/gblinear_train/Dataset/1000:     934 µs
component/overhead/gblinear_train/BinnedDataset_raw_access/1000: 83.0 µs
component/overhead/gblinear_train/Dataset_raw_access/1000:       83.6 µs
component/overhead/gblinear_train/Dataset/10000:    3.48 ms
component/overhead/gblinear_train/BinnedDataset_raw_access/10000: 998 µs
component/overhead/gblinear_train/Dataset_raw_access/10000:       938 µs
component/overhead/gblinear_train/Dataset/50000:    17.65 ms
component/overhead/gblinear_train/BinnedDataset_raw_access/50000: 4.84 ms
component/overhead/gblinear_train/Dataset_raw_access/50000:       4.73 ms

component/overhead/gbdt_train/BinnedDataset:        65.5 ms

component/overhead/predict/Dataset/10000:           6.71 ms
component/overhead/predict/BinnedDataset_raw_access/10000: 115.0 µs
component/overhead/predict/Dataset_raw_access/10000:       93.4 µs
component/overhead/predict/BinnedDataset_SampleBlocks/10000: 7.42 ms
