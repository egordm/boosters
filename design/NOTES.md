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

#### 2024-12: SIMD Row-Parallel Traversal Analysis

Implemented `SimdTraversal` using `wide` crate (f32x8) to process 8 rows in parallel.

**Comprehensive Benchmark (100 trees, 50 features):**

| Strategy | 1K No-Block | 1K Block64 | 10K No-Block | 10K Block64 |
|----------|-------------|------------|--------------|-------------|
| Standard | 2.13ms | 2.14ms | 21.1ms | 21.4ms |
| Unrolled | 793µs | 752µs | 10.2ms | **7.59ms** |
| SIMD | 845µs | 842µs | 8.72ms | 8.41ms |

**Key Findings:**

1. **Standard + blocking = no benefit** (-1.6% at 10K due to overhead)
2. **Unrolled + blocking = huge benefit** (+35% at 10K rows!)
3. **SIMD + blocking = minor benefit** (+3.7%)
4. **SIMD is still 11% slower than Unrolled+Block** (8.41ms vs 7.59ms)

**Why Unrolled+Block is fastest:**

- Level-by-level processing keeps split data in L1/L2 cache
- All rows in a 64-row block traverse same tree level together
- Memory prefetching works optimally with predictable access pattern

**Why SIMD doesn't help more:**

1. **No hardware gather** - `wide` doesn't expose AVX2 `vpgatherdd` instructions
2. **Scalar feature gather** - Must manually extract indices and gather feature values in loops
3. **Scalar NaN handling** - Must check NaN individually per lane
4. **Memory access pattern** - Features stored row-major, but we need column values

**Better approaches (future work):**

- **Column-major (transposed) features** - Would allow vectorized loads
- **Multiple trees in parallel** - Same row through 8 trees simultaneously
- **AVX2 intrinsics** - Use `_mm256_i32gather_ps` for hardware gather
- **std::simd (nightly)** - Has scatter/gather support

**Conclusion:** `UnrolledTraversal` with default block size (64) is the best option.
SIMD row-parallel approach doesn't help with row-major data layout.

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

### YYYY-MM-DD

<!-- What you worked on, what you learned, what's next -->
