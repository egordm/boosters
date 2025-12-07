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
