# RFC-0018: Native Categorical Feature Support

- **Status**: Implemented
- **Created**: 2025-12-18
- **Updated**: 2025-01-21
- **Depends on**: RFC-0004 (Quantization and Binning)
- **Scope**: Native categorical feature handling with optimal partitioning

## Summary

Add native support for categorical features, enabling optimal split finding without
one-hot encoding. This eliminates the 10-100× overhead of one-hot encoding and
improves model quality on high-cardinality categoricals.

## Motivation

### Current State

boosters currently handles categorical features via one-hot encoding:

```text
color ∈ {red, blue, green, yellow}  →  4 binary features
```

This approach has severe limitations:

1. **Memory explosion**: 100 categories → 100 features → 100× histogram memory
2. **Compute overhead**: Build/search 100 histograms vs 1
3. **Quality degradation**: Splitting on one-hot features treats categories independently,
   missing opportunities to group similar categories
4. **Feature bundling can't fully recover**: While RFC-0017 bundles one-hot features,
   optimal categorical splits still require special handling

### Target Use Cases

| Dataset | Categoricals | Cardinality | Current Overhead |
|---------|-------------|-------------|------------------|
| Adult | 8 | 2-42 | 105 one-hot features |
| Criteo | 26 | 10K-10M | Millions of features |
| Avazu | 22 | 100-1M | Millions of features |

### Relationship to Feature Bundling (RFC-0017)

| Scenario | RFC-0017 (Bundling) | RFC-0018 (Native Categorical) |
|----------|---------------------|-------------------------------|
| Pre-encoded one-hot | Bundles back to ~1 feature | N/A (already expanded) |
| Raw categorical | N/A (not applicable) | Handles directly |
| High-cardinality | Limited (max 256 bins) | Optimal partitioning |

**Key insight**: Bundling recovers memory/speed for pre-encoded data, but native
categorical support is needed for optimal split quality and handling raw categoricals.

### When to Use Native Categoricals

**Use native categoricals (this RFC) when:**

- You have raw categorical data (strings or integer IDs)
- High cardinality (>10 unique values)
- You want optimal split quality (grouping similar categories)
- Fresh training without legacy model constraints

**Use one-hot encoding + bundling (RFC-0017) when:**

- Data is already one-hot encoded (preprocessing pipeline)
- Very low cardinality (<5 values per feature)
- You need XGBoost model compatibility (loading existing models)
- Existing pipeline cannot be modified

---

## Design

### Overview

Native categorical support requires changes at three layers:

1. **Data Layer**: Mark features as categorical, store category mappings
2. **Histogram Layer**: Use category-aware histogram accumulation
3. **Split Finding Layer**: Optimal category partitioning algorithm

### Data Layer Changes

#### BinMapper Extensions

```rust
/// Extended BinMapper for categorical features.
impl BinMapper {
    /// Create a categorical bin mapper from raw category values.
    pub fn from_categories(
        categories: &[i32],
        config: CategoricalConfig,
    ) -> Self;
    
    /// Get the category value for a bin (categorical only).
    pub fn bin_to_category(&self, bin: u32) -> Option<i32>;
    
    /// Get the bin for a category value (categorical only).
    pub fn category_to_bin(&self, category: i32) -> Option<u32>;
}

/// Configuration for categorical feature handling.
#[derive(Clone, Debug)]
pub struct CategoricalConfig {
    /// Maximum categories before using partition-based splits (default: 32).
    /// Below this: one-vs-rest splits. Above: optimal partitioning.
    pub max_cat_one_hot: u32,
    
    /// Maximum categories to consider in partition search (default: 64).
    pub max_cat_threshold: u32,
    
    /// Smoothing factor for category statistics (default: 10.0).
    /// Used in: score = gradient / (hessian + cat_smooth)
    pub cat_smooth: f32,
    
    /// Additional L2 regularization for categorical splits (default: 10.0).
    pub cat_l2: f32,
    
    /// Minimum samples per category (default: 10).
    /// Categories with fewer samples are grouped into "other".
    pub min_data_per_group: u32,
}

impl Default for CategoricalConfig {
    fn default() -> Self {
        Self {
            max_cat_one_hot: 32,
            max_cat_threshold: 64,
            cat_smooth: 10.0,
            cat_l2: 10.0,
            min_data_per_group: 10,
        }
    }
}
```

#### FeatureMeta Extensions

```rust
/// Extended feature metadata.
impl FeatureMeta {
    /// Check if this feature is categorical.
    pub fn is_categorical(&self) -> bool;
    
    /// Get categorical configuration (if categorical).
    pub fn categorical_config(&self) -> Option<&CategoricalConfig>;
}
```

#### BinnedDatasetBuilder Extensions

```rust
impl BinnedDatasetBuilder {
    /// Add a categorical feature from raw integer categories.
    pub fn add_categorical(
        self,
        categories: Vec<i32>,
        config: CategoricalConfig,
    ) -> Self;
    
    /// Add a categorical feature with name.
    pub fn add_categorical_named(
        self,
        name: impl Into<String>,
        categories: Vec<i32>,
        config: CategoricalConfig,
    ) -> Self;
    
    /// Specify which columns in a matrix are categorical.
    pub fn with_categorical_columns(
        self,
        categorical_indices: Vec<usize>,
    ) -> Self;
}
```

### Histogram Layer

Categorical features use the same histogram structure (gradient/hessian sums per bin),
but the interpretation differs:

- **Numerical**: Bins are ordered; splits are `feature <= threshold`
- **Categorical**: Bins are unordered; splits are `feature ∈ {subset}`

No changes to `HistogramPool` or `build_histogram` are needed.

### Split Finding Layer

#### Split Types

```rust
/// Type of split condition.
#[derive(Clone, Debug)]
pub enum SplitCondition {
    /// Numerical: go left if value <= threshold.
    Numerical { threshold: f64 },
    
    /// Categorical one-vs-rest: go left if value == category.
    CategoricalOneHot { category: i32 },
    
    /// Categorical partition: go left if value ∈ categories.
    CategoricalPartition { categories: CatBitset },
}

/// Bitset for categorical split representation.
/// Stores which categories go to the left child.
#[derive(Clone, Debug)]
pub struct CatBitset {
    /// Bits packed into u32 chunks.
    bits: Box<[u32]>,
    /// Number of valid categories.
    n_categories: u32,
}

impl CatBitset {
    /// Create empty bitset for n categories.
    pub fn new(n_categories: u32) -> Self;
    
    /// Check if category goes left.
    pub fn contains(&self, category: u32) -> bool;
    
    /// Set category to go left.
    pub fn insert(&mut self, category: u32);
    
    /// Iterate over categories that go left.
    pub fn iter_left(&self) -> impl Iterator<Item = u32>;
}
```

#### Split Finding Algorithm

```text
ALGORITHM: FindBestCategoricalSplit(histogram, config)
------------------------------------------------------
INPUT:
  histogram: [(grad_sum, hess_sum)] for each category bin
  config: CategoricalConfig
OUTPUT:
  best_split: SplitCondition

n_cats = histogram.len()

// Step 1: Filter categories with sufficient samples
valid_cats = []
FOR cat IN 0..n_cats:
    IF histogram[cat].hess_sum >= config.min_data_per_group:
        valid_cats.push(cat)

// Step 2: Choose algorithm based on cardinality
IF valid_cats.len() <= config.max_cat_one_hot:
    RETURN OneVsRestSplit(histogram, valid_cats)
ELSE:
    RETURN PartitionBasedSplit(histogram, valid_cats, config)
```

#### One-vs-Rest Algorithm

```text
ALGORITHM: OneVsRestSplit(histogram, valid_cats)
------------------------------------------------
For low-cardinality categoricals (≤32 categories).
Tests each category against all others.

best_gain = -infinity
best_cat = None

FOR cat IN valid_cats:
    // Left = this category only
    grad_left = histogram[cat].grad_sum
    hess_left = histogram[cat].hess_sum
    
    // Right = all other categories
    grad_right = total_grad - grad_left
    hess_right = total_hess - hess_left
    
    gain = ComputeGain(grad_left, hess_left, grad_right, hess_right)
    IF gain > best_gain:
        best_gain = gain
        best_cat = cat

RETURN SplitCondition::CategoricalOneHot { category: bin_to_cat(best_cat) }
```

#### Partition-Based Algorithm (LightGBM-style)

```text
ALGORITHM: PartitionBasedSplit(histogram, valid_cats, config)
-------------------------------------------------------------
For high-cardinality categoricals. Sorts by gradient/hessian ratio
and searches for optimal partition.

// Step 1: Compute scores for each category
// The score g/(h+smooth) approximates the optimal leaf value for a category.
// Categories with similar scores should go to the same child, so sorting
// by this metric naturally groups categories that would benefit from the
// same leaf prediction. This is the key insight that makes O(n log n)
// partitioning work instead of exponential enumeration.
scores = []
FOR cat IN valid_cats:
    g = histogram[cat].grad_sum
    h = histogram[cat].hess_sum
    score = g / (h + config.cat_smooth)  // Approximates optimal leaf value
    scores.push((cat, score))

// Step 2: Sort by score (stable sort with category index as tie-breaker
// to ensure deterministic partitioning across runs)
sorted_cats = scores.stable_sorted_by(|(cat, score)| (score, cat))

// Step 3: Search in both directions
best_gain = -infinity
best_partition = None

FOR direction IN [Forward, Backward]:
    left_g, left_h = 0.0, 0.0
    partition = CatBitset::new(n_cats)
    
    FOR i IN 0..min(sorted_cats.len(), config.max_cat_threshold):
        cat = IF direction == Forward THEN sorted_cats[i] ELSE sorted_cats[len-1-i]
        
        left_g += histogram[cat].grad_sum
        left_h += histogram[cat].hess_sum
        partition.insert(cat)
        
        right_g = total_grad - left_g
        right_h = total_hess - left_h
        
        // Apply extra L2 regularization for categoricals
        gain = ComputeGain(left_g, left_h + config.cat_l2, right_g, right_h + config.cat_l2)
        
        IF gain > best_gain:
            best_gain = gain
            best_partition = partition.clone()

RETURN SplitCondition::CategoricalPartition { categories: best_partition }
```

### Tree Node Changes

```rust
/// Extended tree node to support categorical splits.
pub struct TreeNode {
    // ... existing fields ...
    
    /// Split condition (replaces simple threshold).
    pub split_condition: SplitCondition,
}

impl TreeNode {
    /// Check if a sample goes left.
    pub fn goes_left(&self, value: f64) -> bool {
        match &self.split_condition {
            SplitCondition::Numerical { threshold } => value <= *threshold,
            SplitCondition::CategoricalOneHot { category } => {
                (value as i32) == *category
            }
            SplitCondition::CategoricalPartition { categories } => {
                categories.contains(value as u32)
            }
        }
    }
}
```

### Integration with Feature Bundling

When both RFC-0017 (bundling) and RFC-0018 (native categoricals) are available:

| User Input | Recommended Handling |
|------------|---------------------|
| Raw categorical column | Use native categorical (this RFC) |
| Pre-encoded one-hot columns | Bundle them (RFC-0017) |
| User specifies "this is categorical" | Use native categorical |
| Unknown (auto-detect) | Heuristic: if <256 unique values and all integers → categorical |

**Bundle → Categorical conversion** (optional future optimization):
If user provides one-hot columns but indicates they came from the same category,
we could reconstruct the original categorical and use native handling.

#### Decision Flowchart

```text
                    ┌─────────────────────────┐
                    │ What is your data?      │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
     ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
     │ Raw categorical │  │ Pre-encoded   │  │ Mixed/Unknown  │
     │ (strings/ints)  │  │ one-hot cols  │  │                │
     └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
             │                   │                   │
             ▼                   ▼                   ▼
     ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
     │ Use RFC-0018   │  │ Use RFC-0017   │  │ Let auto-detect│
     │ (Native Cat)   │  │ (Bundling)     │  │ handle it      │
     └────────────────┘  └────────────────┘  └────────────────┘
```

**Rationale**:
- Native categoricals provide optimal partitioning for raw data
- Bundling is for pre-processed data where one-hot is already done
- Auto-detection uses heuristics but can be overridden

---

## Design Decisions

### DD-1: Sorting Metric for Partitioning

**Context**: How to order categories for partition search.

**Options**:
1. Gradient / Hessian (XGBoost style)
2. Gradient / (Hessian + smooth) (LightGBM style)
3. Mean target value

**Decision**: Gradient / (Hessian + smooth) (Option 2).

**Rationale**: Matches LightGBM, provides stability for low-count categories.

### DD-2: Split Representation

**Context**: How to store which categories go left.

**Options**:
1. Bitset (1 bit per possible category) - XGBoost style
2. List of category values - LightGBM style
3. Hybrid: list for small, bitset for large

**Decision**: Bitset (Option 1).

**Rationale**: 
- O(1) lookup during inference
- Compact for dense category ranges
- Max overhead: 128KB for 1M categories (acceptable)

### DD-3: Default Configuration

**Context**: What defaults match production-quality behavior.

**Decision**: Match LightGBM defaults.

```rust
CategoricalConfig {
    max_cat_one_hot: 32,
    max_cat_threshold: 64,
    cat_smooth: 10.0,
    cat_l2: 10.0,
    min_data_per_group: 10,
}
```

### DD-4: Category Encoding

**Context**: How to handle category values internally.

**Options**:
1. Use raw category values (i32)
2. Remap to contiguous 0..n indices
3. String → integer mapping

**Decision**: Remap to contiguous indices (Option 2).

**Rationale**:
- Enables bitset representation
- Reduces storage overhead
- String mapping is user's responsibility

### DD-5: Missing/Unknown Categories

**Context**: How to handle missing values and unseen categories at inference.

**Decision**:

- Missing values (NaN) → go to default direction based on training
- Unknown categories → bin 0 (grouped with rare categories)
- Rare categories (<min_data_per_group) → bin 0

**Warning/Logging**: When an unknown category is encountered during inference,
the system should log a warning (at DEBUG level) to help users diagnose 
prediction issues. This can be controlled via a configuration flag:

```rust
pub struct InferenceConfig {
    /// Log warnings when unknown categories are encountered.
    /// Default: true for debug builds, false for release.
    pub warn_unknown_categories: bool,
    
    /// Maximum unique unknown categories to log per feature (prevents spam).
    /// After this limit, a summary "N more unknown categories suppressed" is logged.
    /// Default: 10
    pub max_unknown_warnings: usize,
}
```

The warning includes the first N unique unknown category values to help debugging:
```text
[WARN] Feature "city": unknown categories encountered: ["Unknown City", "NewTown", ...]
       (mapped to bin 0). 5 more unique unknowns suppressed.
```

**Rationale**: Silent failures are hard to debug. Users should be able to 
detect when their inference data contains categories not seen during training.

### DD-6: Categorical L2 Regularization

**Context**: The `cat_l2` parameter adds extra L2 regularization for categorical splits.
How does it interact with the global `lambda` (L2) parameter?

**Decision**: `cat_l2` is **additive** to the global lambda.

```rust
// For numerical splits:
gain = ComputeGain(left_g, left_h + lambda, right_g, right_h + lambda)

// For categorical splits:
gain = ComputeGain(left_g, left_h + lambda + cat_l2, right_g, right_h + lambda + cat_l2)
```

**Rationale**: Categorical splits have more degrees of freedom (can partition into
arbitrary subsets), so they benefit from extra regularization to prevent overfitting
to rare category combinations.

**Tuning note**: When doing hyperparameter search, tune `cat_l2` and `cat_smooth`
together with the global `lambda` (L2) and `alpha` (L1) parameters. These interact
to control overall model complexity.

---

## API

### User-Facing API

```rust
use boosters::{BinnedDatasetBuilder, CategoricalConfig, GbdtParams};

// Method 1: Explicit categorical columns
let dataset = BinnedDatasetBuilder::from_matrix(&matrix, 256)
    .with_categorical_columns(vec![0, 3, 7])  // Columns 0, 3, 7 are categorical
    .build()?;

// Method 2: Add categorical features manually
let dataset = BinnedDatasetBuilder::new()
    .add_categorical(color_values, CategoricalConfig::default())
    .add_categorical_named("city", city_values, CategoricalConfig::default())
    .add_numeric(age_bins, age_mapper)
    .build()?;

// Method 3: String categories (convenience helper)
let cities = vec!["NYC", "LA", "Chicago", "NYC", "LA"];
let dataset = BinnedDatasetBuilder::new()
    .add_categorical_from_strings("city", &cities, CategoricalConfig::default())
    .build()?;

// Training uses categorical info automatically
let params = GbdtParams::default()
    .with_categorical_config(CategoricalConfig {
        max_cat_one_hot: 16,  // Override default
        ..Default::default()
    });

let model = train_gbdt(&dataset, &params)?;

// Check if model uses categoricals (useful for pipeline setup)
if model.has_categorical_features() {
    println!("Model uses {} categorical features", model.categorical_feature_count());
}
```

### Model Introspection

```rust
impl GbdtModel {
    /// Check if any feature uses categorical splits.
    pub fn has_categorical_features(&self) -> bool;
    
    /// Count of features that use categorical splits.
    pub fn categorical_feature_count(&self) -> usize;
    
    /// Indices of categorical features.
    pub fn categorical_feature_indices(&self) -> Vec<usize>;
}
```

### Model Export

Categorical splits must be preserved in exported models:

```rust
/// Tree split in exported model.
pub struct ExportedSplit {
    pub feature_index: u32,
    pub split_type: ExportedSplitType,
}

pub enum ExportedSplitType {
    Numerical { threshold: f64 },
    CategoricalOneHot { category: i32 },
    CategoricalPartition { left_categories: Vec<i32> },
}
```

---

## Integration Points

| Component | Changes Required |
|-----------|------------------|
| `BinMapper` | Add category mapping methods |
| `FeatureMeta` | Add `is_categorical`, config |
| `BinnedDatasetBuilder` | Add categorical feature methods |
| `SplitCondition` | New enum replacing threshold-only |
| `TreeNode` | Use `SplitCondition` |
| `SplitFinder` | Add categorical split algorithms |
| `Inference` | Handle categorical split conditions |
| `Model Export` | Include categorical split data |

---

## Performance Considerations

### Memory

| Cardinality | Bitset Size | Histogram Size |
|-------------|-------------|----------------|
| 32 | 4 bytes | 256 bytes |
| 256 | 32 bytes | 2 KB |
| 1,000 | 128 bytes | 8 KB |
| 10,000 | 1.25 KB | 80 KB |
| 100,000 | 12.5 KB | 800 KB |

For very high cardinality (>100K), consider limiting `max_cat_threshold`.

### Compute

- One-vs-rest: O(n_categories) per node
- Partition-based: O(n_categories × log(n_categories)) for sorting + O(max_cat_threshold) for search

---

## Open Questions

1. **Auto-detection of categoricals?**
   - Could infer from data: all integers, <256 unique values
   - Risk: false positives (e.g., year, zip code)
   - Current: require explicit specification

2. **String category support?**
   - Currently: user must map strings → integers
   - Future: optional string → integer mapping in builder via `StringEncoder`
   - `StringEncoder::encode(&self, s: &str) -> Option<u32>` returns `None` for unknown
   - During inference, `None` maps to bin 0 (unknown/rare category bin)

3. **Interaction with GOSS/SHAP?**
   - GOSS: Works normally (samples rows, not categories)
   - SHAP: Need categorical-aware contribution calculation

---

## Future Work

- [ ] String category support in builder
- [ ] Auto-detection heuristics (opt-in)
- [ ] Category importance tracking
- [ ] SHAP integration for categoricals
- [ ] Ordered categorical support (ordinal features)

---

## References

- LightGBM source: `src/io/bin.cpp`, `src/treelearner/feature_histogram.cpp`
- XGBoost source: `src/tree/split_evaluator.h`, `src/common/bitfield.h`
- [LightGBM Categorical Features](https://lightgbm.readthedocs.io/en/latest/Features.html#optimal-split-for-categorical-features)
- RFC-0017: Feature Bundling (related)
- RFC-0004: Quantization and Binning (extended by this RFC)

---

## Changelog

- 2025-12-18: Initial draft
- 2025-12-18: Persona Review Round 1 - Added explanation for CTR-like sorting in
  partition algorithm, added DD-6 (cat_l2 is additive), added decision flowchart
  for bundling vs categoricals, added string category helper method
- 2025-12-18: Persona Review Round 2 - Added tie-breaker (category index) for 
  deterministic partition sorting, clarified StringEncoder returns Option<u32>
  for unknown strings, expanded DD-5 with unknown category warning/logging
- 2025-12-18: Persona Review Round 3 - Added max_unknown_warnings threshold to
  prevent log spam, logging includes first N unknown category values, added 
  Model introspection methods (has_categorical_features, categorical_feature_count)
- 2025-12-18: Persona Review Round 4 (Final) - Added "When to Use" decision helper
  section in Motivation, added tuning note for cat_l2/cat_smooth with global lambda
- 2025-12-19: **Implementation Complete** - Native categorical features fully working
  
  **DD-7 [DECIDED]**: SoA design chosen over SplitCondition enum.
  - Tree stores `split_types: Box<[SplitType]>` + `CategoriesStorage` separately
  - More cache-friendly for inference traversal (hot path reads split_types linearly)
  - Matches XGBoost's internal representation
  - `SplitType` enum: `Numerical`, `Categorical` (not per-node SplitCondition)
  
  **Key Implementation Locations**:
  - `repr/gbdt/node.rs`: SplitType enum (Numerical, Categorical)
  - `repr/gbdt/categories.rs`: CategoriesStorage (per-tree packed bitsets)
  - `training/gbdt/categorical.rs`: CatBitset (64-bit inline + overflow)
  - `training/gbdt/split/find.rs`: find_onehot_split(), find_sorted_split()
  - `inference/gbdt/traversal.rs`: Categorical split handling
  
  **Implementation Deviations from RFC**:
  - DD-8 [DECIDED]: cat_l2 regularization uses global lambda parameter, not separate.
    Quality benchmarks show parity with LightGBM/XGBoost on Adult (86.57% acc) and
    Covertype (77.12% acc), validating this simpler approach.
  - DD-9 [DECIDED]: Unknown category warning not implemented (requires log dependency).
    Deferred as nice-to-have for post-1.0.
  - DD-10 [DECIDED]: String category support not implemented. Users provide integer IDs.
  - max_cat_one_hot threshold uses `GreedySplitter` config, not separate CategoricalConfig.
  
  **Test Coverage**:
  - 14 categorical-specific tests in src/training/gbdt/categorical.rs
  - Integration test: `tests/training/gbdt.rs::train_with_categorical_features_produces_categorical_splits`
  - Quality benchmark validates accuracy parity with reference implementations
  
  **Quality Results (Adult dataset)**:
  | Library | Accuracy | AUC |
  |---------|----------|-----|
  | boosters | 86.57% | 0.926+ |
  | LightGBM | 86.6% | 0.927 |
  | XGBoost | 86.5% | 0.926 |
- 2025-01-21: Terminology update — standardized header format
