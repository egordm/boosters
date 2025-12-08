# Implementation Notes

Scratch space for thoughts, questions, and observations during implementation.

Periodically review and promote important items to:

- RFCs (design decisions)
- `design/analysis/` (research findings)
- Code comments (implementation details)
- GitHub issues (bugs, TODOs)

---

## Open Questions

Things to figure out or decide later.

- [ ] _Example: Should `SoATreeStorage` store node count or compute from array length?_

---

## Ideas & Observations

Thoughts that might be useful later.

### Performance

#### 2024-11-28: SIMD Row-Parallel Traversal Analysis

Implemented `SimdTraversal` using `wide` crate (f32x8) to process 8 rows in parallel.

**Comprehensive Benchmark (100 trees, 50 features, depth 6):**

| Strategy | 1K No-Block | 1K Block64 | 10K No-Block | 10K Block64 |
|----------|-------------|------------|--------------|-------------|
| Standard | 2.13ms | 2.14ms | 21.1ms | 21.4ms |
| Unrolled | 793µs | 752µs | 10.2ms | **7.59ms** |
| SIMD (wide) | 845µs | 842µs | 8.72ms | 8.41ms |

**Key Findings:**

1. **Standard + blocking = no benefit** (-1.6% at 10K due to overhead)
2. **Unrolled + blocking = huge benefit** (+35% at 10K rows!)
3. **SIMD + blocking = minor benefit** (+3.7%)
4. **SIMD is 11% slower than Unrolled+Block** (8.41ms vs 7.59ms)

#### 2024-11-28: Nightly std::simd Gather Experiment

Tested whether `std::simd` hardware gather (`Simd::gather_or`) would help.

| Strategy | 10K (no AVX2) | 10K (AVX2 native) |
|----------|---------------|-------------------|
| Unrolled+Block64 | 7.56ms | 7.57ms |
| Nightly SIMD gather | 27.5ms | 14.5ms |

**Results:** Even with hardware gather, nightly SIMD is **1.9x slower** than Unrolled.

**Why gather doesn't help:**
- AVX2 `vpgatherdd` has ~10-20 cycle latency per 8 elements
- Random memory access defeats cache prefetching
- Tree traversal is **latency-bound**, not throughput-bound
- Overhead of SIMD setup/teardown per level exceeds scalar loop cost

#### 2024-11-28: XGBoost C++ Analysis

Analyzed XGBoost source - they also do **NOT use SIMD** for prediction.

| Technique | XGBoost | booste-rs |
|-----------|---------|-----------|
| Explicit SIMD | ❌ No | ❌ No (doesn't help) |
| Array tree layout | ✅ Yes (6 levels) | ✅ Yes |
| Block processing | ✅ Yes (64 rows) | ✅ Yes |
| OpenMP threading | ✅ Yes | ❌ Not yet |

**Conclusion:** SIMD row-parallel doesn't work for tree traversal. Focus on:
1. ✅ Unrolled traversal (done)
2. ✅ Block processing (done)
3. 🔜 Thread parallelism (next priority)

<!-- Notes about performance observations, potential optimizations -->

### API Design

<!-- Thoughts about API ergonomics, naming, etc. -->

### XGBoost Behavior

<!-- Observations about XGBoost quirks, edge cases, undocumented behavior -->

---

## Implementation Blockers

Things that are blocking progress.

<!-- Use format: What's blocked | Why | Potential solutions -->

---

## Decisions Made (Not in RFC)

Small decisions that don't warrant RFC updates but should be documented.

<!-- Use format: Decision | Rationale | Date -->

| Decision | Rationale | Date |
|----------|-----------|------|
| _Example: Use `Box<[T]>` over `Vec<T>` for immutable arrays_ | _Signals immutability, slightly smaller_ | _2024-11-27_ |

---

## Code Snippets

Useful code patterns or experiments to reference later.

```rust
// Example: fast NaN check
fn has_nan(slice: &[f32]) -> bool {
    slice.iter().any(|x| x.is_nan())
}
```

---

## Links & References

Useful resources discovered during implementation.

- [XGBoost model format docs](https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html)
- [bincode usage examples](https://docs.rs/bincode/latest/bincode/)

---

## Session Log

Brief notes from implementation sessions (optional).

### 2024-12-02: Refactoring Analysis

**File Size Analysis** - Files > 600 lines:

| File | Lines | Assessment |
|------|-------|------------|
| `trainer.rs` (gbtree) | 1960 | Split tests + params |
| `quantize.rs` | 1328 | Split into cuts/matrix modules |
| `split.rs` | 1261 | Keep: cohesive |
| `grower.rs` | 1076 | Already modular |
| `dense.rs` | 1067 | Split iterators |
| `metric.rs` | 958 | Split by metric type |
| `trainer.rs` (linear) | 897 | Split tests |

**Trainer API Analysis:**

Current:
- `train(quantized, ...)` - requires pre-processing
- `train_with_data(data, ...)` - handles quantization
- `train_multiclass(...)` - separate method

Proposed:
- `train(data, labels)` - simple default API
- `train_quantized(...)` - advanced with pre-processed data
- Keep `multiclass` terminology (standard in ML)

**Deduplication Strategy:**

Create `train_internal<M: TrainMode>()` that handles both single/multi-output
via an enum, with thin public wrappers.

Key differences to parameterize:
- `num_outputs`: 1 vs K
- Trees per round: 1 vs K  
- Gradient view: direct vs strided

**Dead Code Check:** Run `cargo +nightly udeps` and `cargo clippy --all-targets`

### YYYY-MM-DD

<!-- What you worked on, what you learned, what's next -->

---

## RFC Prioritization Analysis (2024-12-02)

### Context

After completing RFC-0025 (Row-Parallel Histograms), we need to prioritize the next RFCs:
- Gradient Quantization (would be RFC-0026)
- Linear Trees (would be RFC-0027)
- GPU Support (would be RFC-0028)
- Sample Weighting (would be RFC-0029)

### Dependency Graph

```text
┌────────────────────────────────────────────────────────────────────┐
│                     RFC Dependencies                                │
│                                                                     │
│  RFC-0025 Row-Parallel Histograms ──────────┐                      │
│           │                                  │                      │
│           ▼                                  ▼                      │
│  RFC-0026 Gradient Quantization     RFC-0028 GPU Support           │
│  (16-bit packed gradients)          (CUDA histogram building)       │
│           │                                  │                      │
│           └──────────────────────────────────┘                      │
│                                                                     │
│  RFC-0029 Sample Weighting ─────────────────────► (Independent)    │
│  (instance weights in loss)                                        │
│                                                                     │
│  RFC-0027 Linear Trees ─────────────────────────► (Independent)    │
│  (linear models in leaves)                                         │
└────────────────────────────────────────────────────────────────────┘
```

### Analysis by RFC

#### RFC-0026: Gradient Quantization (16-bit Packed)

**What**: LightGBM's optimization — pack (gradient, hessian) into 16-bit integers
instead of (f32, f32). Reduces memory bandwidth during histogram building by 4x.

**Dependencies**:
- Builds on RFC-0025 (SoA layout already supports this)
- Orthogonal to row-parallel vs feature-parallel

**Value**:
- 📊 High impact on memory-bound workloads
- 🔄 Required for LightGBM training parity
- ⚡ 2-4x speedup on histogram building (memory bandwidth limited)

**Complexity**: Medium
- Quantization mapping for gradients
- Modified accumulate kernel (SIMD pack/unpack)
- Loss of precision (generally acceptable for GBDT)

**When to implement**: After RFC-0025, before GPU

---

#### RFC-0027: Linear Trees

**What**: Instead of constant leaf values, fit a linear model in each leaf.
Captures local trends within regions, improving accuracy on smooth functions.

**Dependencies**:
- Requires changes to `LeafValue` trait (currently `ScalarLeaf`)
- New split evaluation with linear fit consideration
- New leaf fitting algorithm (mini OLS in each leaf)

**Value**:
- 📊 Medium impact — improves accuracy on specific problems
- 🔬 Research feature (not in XGBoost, limited LightGBM support)
- 🎯 Niche use case

**Complexity**: High
- New `LinearLeaf` type with coefficients
- Modified split finding (include linear fit residual)
- Regularization for leaf coefficients
- Prediction path changes

**When to implement**: Post-1.0, based on user demand

---

#### RFC-0028: GPU Support (CUDA)

**What**: GPU-accelerated histogram building and split finding.
This is the biggest performance win for large datasets.

**Dependencies**:
- RFC-0025 (contiguous histogram pool is GPU-friendly)
- RFC-0026 (quantized gradients reduce GPU memory bandwidth)
- Requires `cuda` or `wgpu` infrastructure

**Value**:
- 📊 Very high impact on large datasets
- ⚡ 10-100x speedup potential
- 🏆 Competitive with GPU-accelerated XGBoost/LightGBM

**Complexity**: Very High
- New `cuda` feature flag and crate
- GPU memory management for histograms, quantized data
- Kernel implementation (histogram building, reduction)
- CPU↔GPU synchronization
- Testing across GPU architectures

**When to implement**: After CPU path is optimized (RFC-0025, RFC-0026)

---

#### RFC-0029: Sample Weighting

**What**: Per-instance weights that affect the loss function.
Common for: class imbalance, importance sampling, survey data.

**Dependencies**:
- Modifies gradient computation (multiply by weight)
- Affects histogram accumulation (weighted counts)
- Independent of parallelism strategy

**Value**:
- 📊 High practical value — common real-world need
- ✅ Required for XGBoost/LightGBM parity
- 🎯 Users expect this feature

**Complexity**: Low-Medium
- `DataMatrix` gains optional `weights: Vec<f32>`
- Objective functions multiply gradients by weight
- Histogram counts become weighted
- Evaluation metrics use weighted averages

**When to implement**: Soon — low complexity, high value

---

### Recommended Priority Order

```text
Priority 1: Sample Weighting (RFC-0026)
════════════════════════════════════════
Rationale:
- Low complexity, high user value
- Unblocks real-world use cases
- Independent, can be done anytime
- Estimated: 1-2 weeks

Priority 1.5: Row-Parallel Histograms (RFC-0025)
════════════════════════════════════════════════
Rationale:
- Medium complexity, high performance value
- Already designed, ready to implement
- Foundation for other optimizations
- Estimated: 2-3 weeks

Priority 2: Gradient Quantization (RFC-0027)
════════════════════════════════════════════
Rationale:
- Medium complexity, high performance value
- Natural extension of RFC-0025 work
- Required for LightGBM parity
- Prerequisite for efficient GPU support
- Estimated: 2-3 weeks

Priority 3: GPU Support (RFC-0028)
════════════════════════════════════
Rationale:
- Very high complexity, very high value
- Requires RFC-0025 + RFC-0027 first
- Major competitive differentiator
- Estimated: 1-2 months

Priority 4: Linear Trees (RFC-0029)
════════════════════════════════════
Rationale:
- High complexity, valuable for smooth functions
- Not niche, but requires solid GBTree foundation first
- Defer until core GBTree training is mature
- Estimated: 2-3 weeks
```

### Alternative Ordering Considerations

**If targeting LightGBM training parity first**:
1. Sample Weighting (required)
2. Gradient Quantization (LightGBM's key optimization)
3. RFC-0016 Categorical Training (already drafted)
4. GPU later

**If targeting XGBoost training parity first**:
1. Sample Weighting (required)
2. RFC-0023 Constraints (monotonic, interaction)
3. RFC-0025 Row-Parallel Histograms (done)
4. GPU later

**If targeting maximum performance first**:
1. RFC-0025 Row-Parallel Histograms (done)
2. Gradient Quantization
3. GPU Support
4. Sample weighting and features later

### Summary Table

| RFC | Complexity | User Value | Perf Impact | Dependencies | Priority |
|-----|------------|------------|-------------|--------------|----------|
| Sample Weighting | Low | ⭐⭐⭐⭐⭐ | Low | None | **P1** |
| Row-Parallel Histograms | Medium | ⭐⭐⭐ | ⭐⭐⭐⭐ | None | **P1.5** |
| Gradient Quantization | Medium | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | RFC-0025 | **P2** |
| GPU Support | Very High | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | RFC-0025, RFC-0027 | **P3** |
| Linear Trees | High | ⭐⭐⭐⭐ | ⭐⭐⭐ | Mature GBTree | **P4** |
