# Epic 8: Row-Parallel Histogram Building

**Status**: Not Started  
**Priority**: High  
**Depends on**: Epic 3 (GBTree Training Phase 1), Epic 5 (Phase 2)  
**RFC**: [RFC-0025](../design/rfcs/0025-row-parallel-histograms.md)

## Overview

Implement row-parallel histogram building with advanced memory pooling to achieve
2-5× speedup on tall datasets (many rows, fewer features). This is a critical
performance optimization that brings us to parity with XGBoost and LightGBM.

**Key insight**: Current per-feature parallelism reads each row's gradients once
per feature. Row-parallel parallelism reads each row only once total, reducing
memory traffic by a factor of `num_features`.

---

## Success Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| **Correctness** | Histogram values match sequential | Exact (within f32 tolerance) |
| **Tall data speedup** | 100K rows × 50 features | 2-4× faster than current |
| **Wide data parity** | 10K rows × 500 features | No regression (< 5%) |
| **Memory efficiency** | Pool eviction under pressure | Training completes |
| **LRU correctness** | Evict least recently used | Verified by test |

---

## Testing Strategy Overview

### Test Categories

| Category | Purpose | When to Run |
|----------|---------|-------------|
| **Unit tests** | Pool operations, scratch buffers, reduction | Every commit |
| **Integration tests** | End-to-end histogram building, tree training | Every commit |
| **Correctness tests** | Compare row-parallel vs sequential | Story completion |
| **Performance tests** | Benchmark speedup vs baseline | Story 5 & 6 |
| **Stress tests** | Deep trees, memory pressure | Story 3 |

### Test Data Shapes

| Name | Rows | Features | Purpose |
|------|------|----------|---------|
| `tiny` | 100 | 5 | Unit tests, fast iteration |
| `tall_narrow` | 100K | 20 | Row-parallel sweet spot |
| `medium` | 10K | 100 | Balanced workload |
| `wide` | 5K | 500 | Feature-parallel sweet spot |
| `large` | 500K | 50 | Stress test |

---

## Story 1: ContiguousHistogramPool ✓

**Goal**: Implement the contiguous memory pool with LRU eviction.

**Status**: Not Started

### Tasks

- [ ] 1.1: Implement `ContiguousHistogramPool` with SoA backing store
- [ ] 1.2: Implement `get_or_allocate(node)` with LRU eviction
- [ ] 1.3: Implement `get(node)` with LRU position update  
- [ ] 1.4: Implement `release(node)` and `release_batch(nodes)`
- [ ] 1.5: Implement `reset()` for new tree
- [ ] 1.6: Implement `PoolMetrics` tracking (hits, misses, evictions)
- [ ] 1.7: Implement `recommended_pool_capacity()` sizing heuristic

### Unit Tests

- [ ] `test_pool_basic_allocation` - Allocate and retrieve histogram
- [ ] `test_pool_lru_eviction` - When full, evicts least recently used
- [ ] `test_pool_lru_order` - Access updates LRU position
- [ ] `test_pool_release` - Explicit release frees slot
- [ ] `test_pool_release_batch` - Batch release frees multiple slots
- [ ] `test_pool_reset` - Reset clears all nodes, keeps allocations
- [ ] `test_pool_metrics` - Tracks hits, misses, evictions correctly
- [ ] `test_pool_full_capacity` - Uses all capacity before evicting
- [ ] `test_pool_contains` - `contains()` returns correct boolean

### Implementation Notes

- SoA layout: separate `Box<[f32]>` for grads, hess, counts
- HashMap<NodeId, SlotId> for O(1) node lookup
- VecDeque for LRU queue (or intrusive list for O(1) removal)
- `slot_to_lru_pos` for O(1) LRU position lookup

---

## Story 2: RowParallelScratch & Reduction

**Goal**: Implement per-thread scratch buffers and histogram reduction.

**Status**: Not Started

### Tasks

- [ ] 2.1: Implement `RowParallelScratch` with per-thread SoA buffers
- [ ] 2.2: Implement `get_buffer(thread_id)` for thread-safe access
- [ ] 2.3: Implement `reset_all()` and `reset(thread_id)`
- [ ] 2.4: Implement `reduce_histograms()` - merge thread buffers into target
- [ ] 2.5: Implement SIMD-optimized reduction (portable_simd or manual)
- [ ] 2.6: Consider hierarchical reduction for >8 threads

### Unit Tests

- [ ] `test_scratch_basic` - Create and access scratch buffers
- [ ] `test_scratch_reset` - Reset zeroes all bins
- [ ] `test_reduce_single_thread` - Reduce with 1 thread = identity
- [ ] `test_reduce_multiple_threads` - Sum across threads correct
- [ ] `test_reduce_empty_threads` - Handle threads with no data
- [ ] `test_reduce_large_histogram` - Stress test with many bins

### Performance Tests

- [ ] `bench_reduce_2_threads` vs `bench_reduce_8_threads`
- [ ] `bench_reduce_linear` vs `bench_reduce_hierarchical` (if implemented)

### Implementation Notes

- Each thread's buffer: `bins_per_hist × (f32 + f32 + u32)` bytes
- Total scratch: `num_threads × bins_per_hist × 12` bytes
- For 8 threads, 100 features, 256 bins: ~2.4MB scratch

---

## Story 3: ParallelHistogramBuilder

**Goal**: Implement the row-parallel histogram building algorithm.

**Status**: Not Started

### Tasks

- [ ] 3.1: Implement `ParallelHistogramBuilder::new(num_threads, bins_per_hist)`
- [ ] 3.2: Implement `build()` - partition rows, build in scratch, reduce
- [ ] 3.3: Implement `build_ordered()` - use pre-reordered gradients
- [ ] 3.4: Implement `build_batch()` - build multiple nodes in one pass
- [ ] 3.5: Extract common accumulation kernel to `kernel.rs`
- [ ] 3.6: Ensure `build()` and `build_parallel()` use same kernel

### Unit Tests

- [ ] `test_row_parallel_matches_sequential` - Same result as `build()`
- [ ] `test_row_parallel_matches_feature_parallel` - Same as `build_parallel()`
- [ ] `test_row_parallel_subset_rows` - Works with row subset
- [ ] `test_row_parallel_single_row` - Edge case: 1 row
- [ ] `test_row_parallel_single_feature` - Edge case: 1 feature
- [ ] `test_row_parallel_empty_rows` - Edge case: 0 rows
- [ ] `test_build_batch_multiple_nodes` - Batch build correctness

### Integration Tests

- [ ] `test_row_parallel_in_tree_training` - Full tree builds correctly
- [ ] `test_row_parallel_with_histogram_subtraction` - Subtraction still works

### Implementation Notes

- Row partitioning: `rows.par_chunks(rows_per_thread)`
- Each chunk accumulates into its thread's scratch
- After parallel region, reduce all scratch into target

---

## Story 4: OrderedGradientBuffer

**Goal**: Implement gradient reordering for cache-friendly access.

**Status**: Not Started

### Tasks

- [ ] 4.1: Implement `OrderedGradientBuffer` struct
- [ ] 4.2: Implement `reorder(rows)` - parallel reorder gradients
- [ ] 4.3: Implement `ordered_grads()` and `ordered_hess()` accessors
- [ ] 4.4: Add threshold heuristic (only reorder if rows > 1000)
- [ ] 4.5: Integrate with `ParallelHistogramBuilder::build_ordered()`

### Unit Tests

- [ ] `test_reorder_basic` - Reordered values match expected
- [ ] `test_reorder_identity` - Sequential rows = no change in values
- [ ] `test_reorder_reverse` - Reverse order works
- [ ] `test_reorder_subset` - Subset of rows works
- [ ] `test_reorder_large` - Stress test with many rows

### Performance Tests

- [ ] `bench_reorder_vs_random_access` - Measure reorder overhead vs benefit
- [ ] `bench_reorder_threshold` - Find optimal threshold

### Implementation Notes

- Reorder cost: O(rows) copies
- Benefit: Sequential memory access in hot loop
- Threshold ~1000 rows based on LightGBM experience

---

## Story 5: Strategy Selection & Integration

**Goal**: Add automatic strategy selection and integrate with TreeGrower.

**Status**: Not Started

### Tasks

- [ ] 5.1: Add `ParallelStrategy` enum to `TreeBuildParams`
- [ ] 5.2: Implement `select_histogram_strategy()` heuristic
- [ ] 5.3: Add `min_rows_for_parallel` parameter
- [ ] 5.4: Integrate `ContiguousHistogramPool` into `TreeGrower`
- [ ] 5.5: Update `build_histogram()` to use strategy selection
- [ ] 5.6: Implement histogram subtraction with pool
- [ ] 5.7: Release parent histograms after children built

### Unit Tests

- [ ] `test_strategy_selection_small_node` - Sequential for <1024 rows
- [ ] `test_strategy_selection_tall_narrow` - Row-parallel for tall data
- [ ] `test_strategy_selection_wide` - Feature-parallel for wide data
- [ ] `test_auto_strategy_produces_correct_histogram` - All strategies correct

### Integration Tests

- [ ] `test_train_with_row_parallel` - Training produces valid model
- [ ] `test_train_with_auto_strategy` - Auto selection works
- [ ] `test_train_deep_tree_with_pool` - Pool eviction works for deep trees
- [ ] `test_pool_release_after_split` - Parent released after children built

### Implementation Notes

- Heuristic: `rows / features > threshold` → row-parallel
- Default `min_rows_for_parallel = 1024`
- Pool capacity: `2 × max_leaves × 1.25`

---

## Story 6: Performance Validation

**Goal**: Benchmark and validate performance improvements.

**Status**: Not Started

### Tasks

- [ ] 6.1: Create `benches/histogram_parallel.rs` benchmark suite
- [ ] 6.2: Benchmark row-parallel vs feature-parallel on tall data
- [ ] 6.3: Benchmark row-parallel vs feature-parallel on wide data
- [ ] 6.4: Benchmark gradient reordering impact
- [ ] 6.5: Benchmark pool overhead vs no-pool baseline
- [ ] 6.6: Profile and optimize hot paths
- [ ] 6.7: Document results in `docs/benchmarks/`

### Benchmarks

| Benchmark | Dataset | Metric |
|-----------|---------|--------|
| `bench_hist_tall_sequential` | 100K×20 | Baseline |
| `bench_hist_tall_feature_parallel` | 100K×20 | Current |
| `bench_hist_tall_row_parallel` | 100K×20 | **Target: 2-4× current** |
| `bench_hist_wide_sequential` | 5K×500 | Baseline |
| `bench_hist_wide_feature_parallel` | 5K×500 | Current |
| `bench_hist_wide_row_parallel` | 5K×500 | **Target: ≥0.95× current** |
| `bench_hist_with_reorder` | 100K×20 | Reorder impact |
| `bench_pool_allocation` | N/A | Pool overhead |
| `bench_reduction_threads` | N/A | Reduction scaling |

### Performance Targets

| Dataset Shape | Current | Target | Rationale |
|---------------|---------|--------|-----------|
| 100K rows × 20 features | 1.0× | 2-4× | Row-parallel sweet spot |
| 100K rows × 100 features | 1.0× | 1.5-2× | Balanced |
| 10K rows × 500 features | 1.0× | ≥0.95× | No regression |
| Full training (100K×50) | 1.0× | 1.5-2× | End-to-end |

---

## Story 7: Documentation & Cleanup

**Goal**: Document the new system and clean up code.

**Status**: Not Started

### Tasks

- [ ] 7.1: Add rustdoc for all new public types
- [ ] 7.2: Add module-level documentation for `histogram/` submodule
- [ ] 7.3: Update `histogram/mod.rs` with architecture overview
- [ ] 7.4: Add `histogram_pool_capacity` parameter documentation
- [ ] 7.5: Add `parallel_strategy` parameter documentation
- [ ] 7.6: Write performance tuning guide
- [ ] 7.7: Update ROADMAP with Epic 8 complete

### Documentation Deliverables

- [ ] `src/training/gbtree/histogram/mod.rs` - Architecture overview
- [ ] `src/training/gbtree/histogram/pool.rs` - Pool usage examples
- [ ] `src/training/gbtree/histogram/parallel_builder.rs` - Strategy selection guide
- [ ] `docs/benchmarks/YYYY-MM-DD-row-parallel-histograms.md` - Performance report

---

## Module Structure

After this epic, the histogram module structure will be:

```
src/training/gbtree/histogram/
├── mod.rs               # Re-exports, architecture overview
├── types.rs             # NodeId, SlotId, PoolMetrics (new)
├── feature.rs           # FeatureHistogram (existing)
├── node.rs              # NodeHistogram (existing)
├── kernel.rs            # NEW: Shared accumulation kernel
├── builder.rs           # Sequential + feature-parallel (uses kernel)
├── parallel_builder.rs  # NEW: Row-parallel builder
├── pool.rs              # NEW: ContiguousHistogramPool
└── scratch.rs           # NEW: RowParallelScratch, reduction
```

---

## Dependencies

```
Story 1 (Pool) ──────────────────────────────────────┐
                                                      │
Story 2 (Scratch & Reduction) ───────────────────────┤
                                                      ├──► Story 5 (Integration)
Story 3 (ParallelHistogramBuilder) ──────────────────┤          │
                                                      │          │
Story 4 (OrderedGradientBuffer) ─────────────────────┘          │
                                                                 │
                                    Story 5 (Integration) ◄──────┘
                                             │
                                             ▼
                                    Story 6 (Performance)
                                             │
                                             ▼
                                    Story 7 (Documentation)
```

**Notes**:
- Stories 1-4 can be worked on in parallel (independent components)
- Story 5 integrates all components
- Story 6 validates performance
- Story 7 wraps up documentation

---

## Definition of Done

- [ ] All unit tests passing
- [ ] Integration tests passing (training produces correct models)
- [ ] Performance targets met:
  - [ ] 2× speedup on tall data (100K×20)
  - [ ] No regression on wide data (5K×500)
- [ ] Pool eviction works under memory pressure
- [ ] No memory leaks (verified with Miri if applicable)
- [ ] Rustdoc complete for public APIs
- [ ] Benchmark report published
- [ ] No compiler warnings
- [ ] `cargo clippy` clean

---

## Open Questions (from RFC-0025)

1. **Hierarchical reduction threshold?**
   - Linear reduction is O(threads × bins)
   - Tree reduction is O(log(threads) × bins) with more overhead
   - Benchmark to find crossover point (likely 8-16 threads)

2. **Gradient reordering threshold?**
   - Reordering is O(rows) extra work
   - Benefit is sequential access in inner loop
   - LightGBM uses ~1000 rows; validate with benchmarks

3. **Auto-strategy heuristic tuning?**
   - Initial: `rows / features > 100` → row-parallel
   - May need dataset-specific tuning

---

## Future Work (Out of Scope)

- [ ] SIMD-optimized accumulation kernel (portable_simd)
- [ ] NUMA-aware thread placement
- [ ] 16-bit packed gradients (RFC-0027)
- [ ] GPU histogram building (separate RFC)
- [ ] Distributed allreduce integration

---

## References

- [RFC-0025: Row-Parallel Histogram Building](../design/rfcs/0025-row-parallel-histograms.md)
- [RFC-0012: Histogram Building](../design/rfcs/0012-histogram-building.md)
- [XGBoost ParallelGHistBuilder](https://github.com/dmlc/xgboost/blob/master/src/common/hist_util.h)
- [LightGBM HistogramPool](https://github.com/microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp)
