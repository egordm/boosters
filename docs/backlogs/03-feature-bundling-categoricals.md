# Backlog: Feature Bundling & Native Categorical Support

**Scope**: RFC-0017 (Feature Bundling), RFC-0018 (Native Categorical Features)  
**Created**: 2025-12-18  
**Status**: Ready for Implementation

---

## Scope & Prioritization

**MVP** (Must Have):

- Epic 1: Stories 1.1-1.4 (Core bundling functionality)
- Epic 2: Stories 2.1-2.4 (Core categorical functionality)

**Post-MVP** (Nice to Have):

- Story 1.5: Advanced diagnostics and presets
- String-to-integer categorical mapping helper
- Bundle-aware inference memory optimization

### Story Dependencies

```text
Epic 1 (Bundling):        Epic 2 (Categorical):

  1.1 ──► 1.2 ──► 1.3       2.1 ──► 2.2
              │   │             │   │
              ▼   ▼             ▼   ▼
            1.4  1.5           2.3  2.4

Cross-epic: Story 2.1 (SplitCondition) should be implemented first
as it introduces the shared type used by training and inference.
```

---

## Module Structure

### Current Structure (relevant modules)

```text
src/
├── data/
│   ├── dataset.rs           # Dataset, FeatureColumn::Categorical
│   └── binned/
│       ├── bin_mapper.rs    # BinMapper (has categorical())
│       ├── builder.rs       # BinnedDatasetBuilder
│       ├── dataset.rs       # BinnedDataset
│       └── group.rs         # FeatureGroup, FeatureMeta
│
├── training/gbdt/
│   ├── categorical.rs       # CatBitset (EXISTS)
│   ├── split/finder.rs      # SplitFinder
│   └── histograms/          # Histogram building
│
├── repr/
│   └── node.rs              # TreeNode
│
└── inference/               # Tree traversal
```

### New Files

```text
src/data/binned/
├── bundling.rs              # BundlingConfig, BundlePlan, algorithms
└── feature_analysis.rs      # FeatureInfo, FeatureStats

src/training/gbdt/split/
└── categorical_split.rs     # One-vs-rest, partition algorithms
```

### Integration Points

| File | Change | Story |
|------|--------|-------|
| `data/binned/builder.rs` | Add bundling pipeline | 1.3 |
| `data/binned/dataset.rs` | Add bundle_plan field | 1.3 |
| `data/binned/group.rs` | Add BundleMeta | 1.3 |
| `repr/node.rs` | SplitCondition enum | 2.1 |
| `training/gbdt/split/finder.rs` | Categorical dispatch | 2.2 |
| `inference/*.rs` | Handle SplitCondition | 2.4 |

---

## Epic 1: Feature Bundling (RFC-0017)

**Goal**: 5-10× memory reduction for sparse/one-hot datasets.

### Story 1.1: Feature Analysis ✅

Single-pass detection of binary and sparse features.

- [x] **1.1.1**: `FeatureInfo` struct (density, is_binary, is_trivial)
- [x] **1.1.2**: Single-pass analysis with O(1) memory per feature
- [x] **1.1.3**: Parallelize over columns with rayon

**DoD**: Detects any 2-value feature as binary (not just {0,1}). ✅

**Implementation**: `src/data/binned/feature_analysis.rs` with 13 unit tests.

---

### Story 1.2: Conflict Detection & Bundling ✅

Build conflict graph and assign features to bundles.

- [x] **1.2.1** (1h): `BundlingConfig` with defaults from RFC
- [x] **1.2.2** (3h): Bitset-based conflict graph (O(n_sparse²) pairs)
- [x] **1.2.3** (1h): Row sampling (≤10K) for large datasets
- [x] **1.2.4** (2h): Greedy bundle assignment
- [x] **1.2.5** (0.5h): Skip if >1000 sparse features (log warning)

**DoD**: Perfect one-hot encoding → 1 bundle per categorical. ✅

**Implementation**: `src/data/binned/bundling.rs` with 20 unit tests.
- `BundlingConfig` with `auto()`, `disabled()`, `aggressive()`, `strict()` presets
- `ConflictGraph` using `fixedbitset` for efficient bitset operations
- `sample_rows()` with stratified sampling (80% random + 10% first + 10% last)
- `assign_bundles()` greedy algorithm sorted by density
- `create_bundle_plan()` main entry point with early termination

---

### Story 1.3: Bundle Encoding & Integration

Encode bundled features and integrate with BinnedDatasetBuilder.

- [ ] **1.3.1** (1h): `BundlePlan`, `BundleMeta`, `FeatureMapping` structs
- [ ] **1.3.2** (2h): Offset-based encoding (bin = offset + feature_bin)
- [ ] **1.3.3** (2h): `BinnedDatasetBuilder::with_bundling()` API
- [ ] **1.3.4** (1h): Store bundle_plan in BinnedDataset
- [ ] **1.3.5** (1h): `bundling_stats()` method

**DoD**: Adult dataset: 105 features → ~14 bundles.

**Tests**:

- Roundtrip: encode → decode returns original (feature, bin)
- Stats: `is_effective()` true when reduction >20%
- Memory: bundled dataset uses <1MB vs ~5MB unbundled (Adult)

---

### Story 1.4: Histogram Integration

Histogram building works correctly with bundled features.

- [ ] **1.4.1** (2h): Histogram builder handles bundle columns
- [ ] **1.4.2** (1h): Split decoding to original features
- [ ] **1.4.3** (1h): Column sampling maps to original features
- [ ] **1.4.4** (0.5h): Validate bundle_hints (error on bad indices)

**DoD**: Bundled training produces same quality as unbundled (±0.002 AUC).

**Tests**:

- Quality: Adult AUC ≥ 0.926 (baseline 0.927)
- Feature importance: reports original feature indices
- Column sampling: 50% samples half of original features

**Histogram note**: Each row contributes to exactly ONE bin per bundle.
No double-counting occurs; conflicts just mean bin represents multiple features.

---

### Story 1.5: User API & Diagnostics (Post-MVP)

User-friendly configuration and visibility.

- [ ] **1.5.1** (0.5h): `auto()`, `disabled()`, `aggressive()` presets
- [ ] **1.5.2** (1h): INFO/DEBUG logging for bundling decisions
- [ ] **1.5.3** (0.5h): `bundling_efficiency()` heuristic metric
- [ ] **1.5.4** (1h): Example in examples/ directory

**DoD**: User can understand bundling behavior from logs.

---

## Epic 2: Native Categorical Features (RFC-0018)

**Goal**: Optimal categorical splits without one-hot encoding.

**Status**: ✅ Core infrastructure complete. Minor verification work remaining.

> **Implementation Note (2025-01-XX)**: Upon exploration, this epic was ~90% 
> implemented. The codebase uses a SoA design (SplitType enum + CategoriesStorage)
> rather than the RFC's proposed SplitCondition enum, which is more cache-friendly.
> See RFC-0018 changelog for design decision update.

### Story 2.1: SplitCondition Infrastructure ✅

Introduce categorical split types to tree representation.

- [x] **2.1.1**: `SplitType` enum in `repr/gbdt/node.rs` (Numeric, Categorical)
- [x] **2.1.2**: `CatBitset` in `training/gbdt/categorical.rs` (64-bit inline + overflow)
- [x] **2.1.3**: `CategoriesStorage` in `repr/gbdt/categories.rs` (per-tree packed bitsets)
- [x] **2.1.4**: `MutableTree` has `apply_categorical_split()` and `set_categorical_split()`

**Already implemented. No additional work required.**

---

### Story 2.2: Categorical Split Finding ✅

Implement one-vs-rest and partition-based split algorithms.

- [x] **2.2.1**: `max_onehot_cats` config (default 4) in `GreedySplitter`
- [x] **2.2.2**: `find_onehot_split()` - O(k) scan for low cardinality
- [x] **2.2.3**: `find_sorted_split()` - CTR sorting for high cardinality
- [x] **2.2.4**: `find_split()` dispatches by `feature_types[f]` boolean
- [ ] **2.2.5** (0.5h): Apply cat_l2 regularization (verify/add if missing)

**Note**: Regularization may use existing lambda. Verify cat_l2 is additive.

---

### Story 2.3: Dataset Integration ✅

Enable categorical column specification in builder API.

- [x] **2.3.1**: `FeatureColumn::Categorical { values: Vec<i32> }` exists
- [x] **2.3.2**: `BinMapper::categorical()` constructor exists
- [x] **2.3.3**: `BinMapper::is_categorical()` method exists
- [x] **2.3.4**: `BinnedDataset::is_categorical(feature)` flows to grower

**DoD**: Categorical designation flows from Dataset to SplitFinder. ✅

---

### Story 2.4: Inference & Introspection ✅

Handle categorical splits during prediction.

- [x] **2.4.1**: `Tree::predict_row()` handles `SplitType::Categorical`
- [x] **2.4.2**: `CategoriesStorage::category_goes_right()` returns false for out-of-range
- [x] **2.4.3**: `TreeView::has_categorical()` method exists
- [ ] **2.4.4** (0.5h): Rate-limited warning for unknown categories (nice-to-have)

**DoD**: Inference matches training behavior exactly. ✅

---

### Story 2.5: End-to-End Verification ✅

Add integration test to verify full pipeline works.

- [x] **2.5.1**: Integration test: train with categorical features, verify splits
- [x] **2.5.2**: Verify predictions match expected values

**DoD**: At least one test trains a model with categorical data and produces
categorical splits. ✅

**Implementation**: `tests/training/gbdt.rs::train_with_categorical_features_produces_categorical_splits`

---

## Testing Strategy

### Test Hierarchy

| Level | Coverage | Location |
|-------|----------|----------|
| Unit | Individual algorithms | `src/**/mod.rs` |
| Integration | Full pipelines | `tests/*.rs` |
| Quality | Accuracy benchmarks | `quality_benchmark` |
| Performance | Speed/memory | `benches/` |

### Quality Acceptance Criteria

| Scenario | Metric | Pass Criterion |
|----------|--------|----------------|
| Adult + bundling | AUC | ≥ 0.926 |
| Adult + native categorical | AUC | ≥ 0.928 |
| Synthetic one-hot (100 features) | Memory | ≤ 20% of unbundled |
| High-cardinality (1K categories) | Training time | ≤ 2× numerical baseline |

### Edge Case Checklist

- [ ] 0 rows, 0 features
- [ ] All-NaN feature
- [ ] Single-value feature (trivial)
- [ ] 1 category in categorical feature
- [ ] Unknown category at inference
- [ ] Bundle of 1 feature
- [ ] All features dense (bundling skipped)
- [ ] >1000 sparse features (bundling skipped)

---

## Definition of Done

**Per Story**:

- [ ] Code compiles (`cargo check`)
- [ ] No clippy warnings (`cargo clippy`)
- [ ] Unit tests pass
- [ ] Integration tests pass (if applicable)
- [ ] Public APIs documented with examples

**Per Epic**:

- [ ] Quality benchmark passes
- [ ] No performance regression (±5%)
- [ ] Example in examples/ directory

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Bundle conflicts degrade quality | Low | Medium | 0.01% default tolerance |
| Categorical overfitting | Medium | Medium | cat_l2=10.0 default |
| Breaking model format | Low | High | Version field in serialization |
| Performance regression | Low | Medium | Benchmark after each story |
| Scope creep | Medium | Medium | MVP clearly defined |

---

## Effort Estimate

| Story | Tasks | Estimated | Status |
|-------|-------|-----------|--------|
| 1.1 Feature Analysis | 3 | 4h | ✅ Complete |
| 1.2 Conflict/Bundling | 5 | 7.5h | ✅ Complete |
| 1.3 Encoding/Integration | 5 | 7h | Not Started |
| 1.4 Histogram Integration | 4 | 4.5h | Not Started |
| 1.5 User API (post-MVP) | 4 | 3h | Post-MVP |
| 2.1 SplitCondition | 4 | 7h | ✅ Complete |
| 2.2 Split Finding | 5 | 8.5h | ✅ Complete |
| 2.3 Dataset Integration | 4 | 3.5h | ✅ Complete |
| 2.4 Inference | 4 | 4.5h | ✅ Complete |
| 2.5 Verification | 2 | 1.5h | ✅ Complete |
| **Remaining MVP** | 9 | ~11.5h | |

---

## Changelog

- 2025-12-18: Initial draft
- 2025-12-18: Round 1 - Added MVP vs post-MVP, story dependencies
- 2025-12-18: Round 2 - Added edge cases, complexity limits, specific tolerances
- 2025-12-18: Round 3 - Simplified structure, added effort estimates
- 2025-12-18: Round 4 - Added architecture notes, histogram clarification
- 2025-12-18: Round 5 - Consolidated testing, removed redundancy
- 2025-12-18: Round 6 - Final polish, ready for review
- 2025-01-XX: Implementation Round 1 - Epic 2 found to be ~90% complete
  - Stories 2.1-2.4 marked complete (SoA design already implemented)
  - Added Story 2.5 (end-to-end verification) with 1.5h estimate
  - Remaining MVP effort reduced from ~46h to ~25h
