# Epic 8: Row-Parallel Histogram Building

**Status**: In Progress  
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

## Story 1: ContiguousHistogramPool ✅ COMPLETE

**Goal**: Implement the contiguous memory pool with LRU eviction.

**Status**: Complete

### Tasks

- [x] 1.1: Implement `ContiguousHistogramPool` with SoA backing store
- [x] 1.2: Implement `get_or_allocate(node)` with LRU eviction
- [x] 1.3: Implement `get(node)` with LRU position update  
- [x] 1.4: Implement `release(node)` and `release_batch(nodes)`
- [x] 1.5: Implement `reset()` for new tree
- [x] 1.6: Implement `PoolMetrics` tracking (hits, misses, evictions)
- [x] 1.7: Implement `recommended_pool_capacity()` sizing heuristic

### Unit Tests

- [x] `test_pool_basic_allocation` - Allocate and retrieve histogram
- [x] `test_pool_lru_eviction` - When full, evicts least recently used
- [x] `test_pool_lru_order` - Access updates LRU position
- [x] `test_pool_release` - Explicit release frees slot
- [x] `test_pool_release_batch` - Batch release frees multiple slots
- [x] `test_pool_reset` - Reset clears all nodes, keeps allocations
- [x] `test_pool_metrics` - Tracks hits, misses, evictions correctly
- [x] `test_pool_full_capacity` - Uses all capacity before evicting
- [x] `test_pool_contains` - `contains()` returns correct boolean

### Implementation Notes

- SoA layout: separate `Box<[f32]>` for grads, hess, counts
- HashMap<NodeId, SlotId> for O(1) node lookup
- VecDeque for LRU queue (or intrusive list for O(1) removal)
- `slot_to_lru_pos` for O(1) LRU position lookup

---

## Story 2: RowParallelScratch & Reduction ✅ COMPLETE

**Goal**: Implement per-thread scratch buffers and histogram reduction.

**Status**: Complete

### Tasks

- [x] 2.1: Implement `RowParallelScratch` with per-thread SoA buffers
- [x] 2.2: Implement `get_buffer(thread_id)` for thread-safe access
- [x] 2.3: Implement `reset_all()` and `reset(thread_id)`
- [x] 2.4: Implement `reduce_histograms()` - merge thread buffers into target
- [x] 2.5: Implement parallel reduction (`reduce_into_parallel()`)
- [x] 2.6: Implement `subtract_histograms()` helper

### Unit Tests

- [x] `test_scratch_basic` - Create and access scratch buffers
- [x] `test_scratch_reset` - Reset zeroes all bins
- [x] `test_reduce_single_thread` - Reduce with 1 thread = identity
- [x] `test_reduce_multiple_threads` - Sum across threads correct
- [x] `test_reduce_empty_threads` - Handle threads with no data
- [x] `test_reduce_large_histogram` - Stress test with many bins
- [x] `test_reduce_parallel_matches_sequential` - Parallel matches sequential
- [x] `test_subtract_histograms` - Subtraction is correct

### Implementation Notes

- Each thread's buffer: `bins_per_hist × (f32 + f32 + u32)` bytes
- Total scratch: `num_threads × bins_per_hist × 12` bytes
- For 8 threads, 100 features, 256 bins: ~2.4MB scratch

---

## Story 3: ParallelHistogramBuilder ✅ COMPLETE

**Goal**: Implement the row-parallel histogram building algorithm.

**Status**: Complete

### Tasks

- [x] 3.1: Implement `ParallelHistogramBuilder::new(config, cuts)`
- [x] 3.2: Implement `build_into_pool()` - partition rows, build in scratch, reduce
- [x] 3.3: Implement `build_into_slot()` - build into arbitrary slot
- [x] 3.4: Implement `build_row_parallel()` - core row-parallel algorithm with rayon::scope()
- [x] 3.5: Implement `ParallelHistogramConfig` with threshold selection
- [x] 3.6: Add `SendSyncPtr<T>` wrapper for raw pointer Send+Sync

### Unit Tests

- [x] `test_parallel_builder_basic` - Basic row-parallel build
- [x] `test_parallel_builder_matches_sequential` - Same result as `HistogramBuilder`
- [x] `test_parallel_builder_subset` - Works with row subset
- [x] `test_parallel_builder_single_thread` - Single thread mode
- [x] `test_parallel_builder_rebuild` - Multiple histograms in pool
- [x] `test_parallel_builder_many_threads` - More threads than rows
- [x] `test_config_should_use_row_parallel` - Threshold selection

### Implementation Notes

- Row partitioning: `rows.par_chunks(rows_per_thread)`
- Each chunk accumulates into its thread's scratch
- After parallel region, reduce all scratch into target

---

## Story 4: Histogram Unification (HistogramLayout + FeatureSlice)

**Goal**: Unify histogram storage so `ContiguousHistogramPool` can be used for all
strategies (sequential, feature-parallel, row-parallel), replacing `NodeHistogram`.

**Status**: In Progress

**Rationale**: See RFC-0025 DD-7. The flat SoA layout is more cache-friendly than
nested `NodeHistogram` → `Vec<FeatureHistogram>` for all use cases. This unification
simplifies the codebase and enables all strategies to benefit from pooling/LRU.

### Tasks

- [x] 4.1: Implement `HistogramLayout` struct with feature offsets
- [x] 4.2: Implement `FeatureSlice<'a>` - immutable view into one feature's bins
- [x] 4.3: Implement `FeatureSliceMut<'a>` - mutable view for accumulation
- [x] 4.4: Add `layout.feature_slice(slot, feat)` and `feature_slice_mut()` methods
- [x] 4.5: Implement `add()` on `FeatureSliceMut` (matching `FeatureHistogram`)
- [x] 4.6: Implement `HistogramBins` trait for generic split finding
- [x] 4.7: Update `SplitFinder` to work with `HistogramBins` trait

### Unit Tests

- [x] `test_histogram_layout_uniform` - Correct offsets for uniform bins
- [x] `test_histogram_layout_offsets` - Correct offsets for various bin counts
- [x] `test_feature_slice_basic` - Read stats from feature slice
- [x] `test_feature_slice_totals` - Total grad/hess/count computation
- [x] `test_feature_slice_mut_add` - Accumulate into feature slice
- [x] `test_feature_slice_mut_reset` - Reset zeroes all bins
- [x] `test_feature_slice_mut_as_immut` - Convert mutable to immutable
- [x] `test_histogram_bins_trait` - Trait works with generic functions

### Remaining Work

- [x] 4.8: Merge `ParallelHistogramBuilder` into unified `HistogramBuilder`
  - Add `build_sequential()` method (current behavior)
  - Add `build_feature_parallel()` method
  - Add `build_row_parallel()` method (from ParallelHistogramBuilder)
  - ~~Add `build()` auto-select method based on heuristics~~ (deferred - not needed yet)
  - Remove separate `ParallelHistogramBuilder` struct
- [ ] 4.9: Remove `NodeHistogram` - pool replaces it entirely
- [ ] 4.10: Integration tests: builder with pool slot, split finder with slices

### Implementation Notes

- `HistogramLayout` stores `feature_offsets: Box<[usize]>` computed from `BinCuts`
- `FeatureSlice` is just `(&[f32], &[f32], &[u32])` - grad, hess, count slices
- `HistogramBins` trait provides `num_bins()` and `bin_stats()` for split finding
- Both `FeatureHistogram` and `FeatureSlice` implement `HistogramBins`
- Key insight: Split finding is now generic over histogram storage type

### Type Decisions

| Type | Decision | Reasoning |
|------|----------|-----------|
| `FeatureHistogram` | **Keep** | Useful as owned, standalone histogram. `FeatureSlice` is borrowed; sometimes ownership is needed (tests, standalone use). Both implement `HistogramBins`. |
| `NodeHistogram` | **Remove** | Only groups `Vec<FeatureHistogram>` with cached totals. Pool + `HistogramLayout` replaces this entirely. Keeping it creates confusion about which to use. |
| `FeatureSlice` | **Keep** | Borrowed view into flat pool storage. Zero-cost abstraction over raw slices. |
| `HistogramBuilder` | **Refactor** | Should build into pool slots. Single code path, no dual storage formats. |

---

## Story 5: OrderedGradientBuffer (Optional Performance)

**Goal**: Implement gradient reordering for cache-friendly access.

**Status**: Not Started

**Note**: This is a performance optimization, not required for correctness.
Can be deferred if integration (Story 6) is higher priority.

### Tasks

- [ ] 5.1: Implement `OrderedGradientBuffer` struct
- [ ] 5.2: Implement `reorder(rows)` - parallel reorder gradients
- [ ] 5.3: Implement `ordered_grads()` and `ordered_hess()` accessors
- [ ] 5.4: Add threshold heuristic (only reorder if rows > 1000)
- [ ] 5.5: Integrate with `ParallelHistogramBuilder::build_ordered()`

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

## Story 6: Strategy Selection & Integration

**Goal**: Add automatic strategy selection and integrate with TreeGrower.

**Status**: Not Started

### Tasks

- [ ] 6.1: Add `ParallelStrategy` enum to `TreeBuildParams`
- [ ] 6.2: Implement `select_histogram_strategy()` heuristic
- [ ] 6.3: Add `min_rows_for_parallel` parameter
- [ ] 6.4: Integrate `ContiguousHistogramPool` into `TreeGrower`
- [ ] 6.5: Update `build_histogram()` to use strategy selection
- [ ] 6.6: Implement histogram subtraction with pool
- [ ] 6.7: Release parent histograms after children built

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

## Story 7: Performance Validation

**Goal**: Benchmark and validate performance improvements.

**Status**: Not Started

### Tasks

- [ ] 7.1: Create `benches/histogram_parallel.rs` benchmark suite
- [ ] 7.2: Benchmark row-parallel vs feature-parallel on tall data
- [ ] 7.3: Benchmark row-parallel vs feature-parallel on wide data
- [ ] 7.4: Benchmark gradient reordering impact
- [ ] 7.5: Benchmark pool overhead vs no-pool baseline
- [ ] 7.6: Profile and optimize hot paths
- [ ] 7.7: Document results in `docs/benchmarks/`

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

## Story 8: Documentation & Cleanup

**Goal**: Document the new system and clean up code.

**Status**: Not Started

### Tasks

- [ ] 8.1: Add rustdoc for all new public types
- [ ] 8.2: Add module-level documentation for `histogram/` submodule
- [ ] 8.3: Update `histogram/mod.rs` with architecture overview
- [ ] 8.4: Add `histogram_pool_capacity` parameter documentation
- [ ] 8.5: Add `parallel_strategy` parameter documentation
- [ ] 8.6: Write performance tuning guide
- [ ] 8.7: Update ROADMAP with Epic 8 complete
- [ ] 8.8: Remove deprecated `NodeHistogram` if fully replaced

### Documentation Deliverables

- [ ] `src/training/gbtree/histogram/mod.rs` - Architecture overview
- [ ] `src/training/gbtree/histogram/pool.rs` - Pool usage examples
- [ ] `src/training/gbtree/histogram/parallel_builder.rs` - Strategy selection guide
- [ ] `docs/benchmarks/YYYY-MM-DD-row-parallel-histograms.md` - Performance report

---

## Module Structure

After this epic, the histogram module structure will be:

```text
src/training/gbtree/histogram/
├── mod.rs               # Re-exports, architecture overview
├── types.rs             # NodeId, SlotId, PoolMetrics, HistogramLayout
├── feature.rs           # FeatureHistogram (owned single-feature histogram)
├── slice.rs             # FeatureSlice, FeatureSliceMut (borrowed views)
├── builder.rs           # Sequential + feature-parallel (builds into pool)
├── parallel_builder.rs  # Row-parallel builder
├── pool.rs              # ContiguousHistogramPool (unified storage)
└── scratch.rs           # RowParallelScratch, reduction
```

**Removed**: `node.rs` (`NodeHistogram`) - replaced by pool + layout

---

## Dependencies

```text
Story 1 (Pool) ──────────────────────────────────────┐
                                                      │
Story 2 (Scratch & Reduction) ───────────────────────┤
                                                      ├──► Story 4 (Unification)
Story 3 (ParallelHistogramBuilder) ──────────────────┤          │
                                                                 │
                                                                 ▼
                                                        Story 6 (Integration)
                                                                 │
                            Story 5 (OrderedGradientBuffer) ────►│ (optional)
                                                                 │
                                                                 ▼
                                                        Story 7 (Performance)
                                                                 │
                                                                 ▼
                                                        Story 8 (Documentation)
```

**Notes**:

- Stories 1-3 are complete ✅
- Story 4 (Unification) enables pool to be used for all strategies
- Story 5 (OrderedGradientBuffer) is optional, pure performance optimization
- Story 6 (Integration) connects pool to TreeGrower
- Story 7 validates performance
- Story 8 wraps up documentation

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
