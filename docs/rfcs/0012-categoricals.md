# RFC-0012: Categorical Features

**Status**: Implemented  
**Created**: 2025-12-20  
**Updated**: 2026-01-02  
**Scope**: Categorical feature handling in GBDT

## Summary

Categorical features use specialized split strategies: one-hot for low
cardinality, sorted partition for high cardinality. Categories are stored
as bitsets during training and in tree nodes.

## Why Native Categoricals?

One-hot encoding high-cardinality categoricals creates:
- Many sparse features
- Fragmented splits
- Memory overhead

Native categorical handling:
- Single split can partition multiple categories
- More expressive partitions
- Direct encoding in tree

## Split Strategies

### One-Hot (Low Cardinality)

For ≤ `max_onehot_cats` categories (default: 4):

```text
Category histogram: [A=100, B=50, C=75, D=25]

Try each as singleton:
  {A} vs {B,C,D} → gain_A
  {B} vs {A,C,D} → gain_B
  ...
Pick best singleton.
```

Simple, O(K) comparisons, works well for few categories.

### Sorted Partition (High Cardinality)

For > `max_onehot_cats` categories:

1. Compute gradient ratio per category: `grad_sum / hess_sum`
2. Sort categories by ratio
3. Scan for optimal partition point

```text
Sorted by ratio: [D, B, C, A]

Scan partitions:
  {D} vs {B,C,A} → gain_1
  {D,B} vs {C,A} → gain_2
  {D,B,C} vs {A} → gain_3
Pick best partition.
```

O(K log K) from sorting. Higher-gradient categories go right (convention).

## CatBitset

Compact storage for category sets:

```rust
pub struct CatBitset {
    bits: u64,                      // Inline: categories 0..63
    overflow: Option<Box<[u64]>>,   // Heap: categories 64+
}

impl CatBitset {
    pub fn contains(&self, cat: u32) -> bool;
    pub fn insert(&mut self, cat: u32);
    pub fn count(&self) -> u32;
    pub fn iter(&self) -> impl Iterator<Item = u32>;
}
```

Why inline first 64?
- Most categoricals have < 64 categories
- Avoids allocation in common case
- Fast bitwise operations

## Split Types

```rust
pub enum SplitType {
    Numerical { bin: u16 },
    Categorical { left_cats: CatBitset },
}

pub struct SplitInfo {
    pub feature: u32,
    pub gain: f32,
    pub default_left: bool,
    pub split_type: SplitType,
}
```

Numerical: samples with bin ≤ threshold go left.
Categorical: samples with category in `left_cats` go left.

## Tree Traversal

At a categorical split node:

```rust
fn traverse_categorical(sample: &[f32], split: &CatBitset) -> bool {
    let category = sample[feature] as u32;
    split.contains(category)  // true = go left
}
```

Missing values follow `default_left`.

## Training Integration

Histogram building treats categorical bins as category indices:

```text
Feature bins: [0, 2, 1, 0, 2]  (categories 0, 1, 2)
Histogram:
  bin 0: (grad_sum=1.2, hess_sum=3.0)
  bin 1: (grad_sum=0.5, hess_sum=1.5)
  bin 2: (grad_sum=0.8, hess_sum=2.0)
```

Split finder detects categorical feature (via feature metadata) and uses
appropriate strategy.

## Feature Metadata

```rust
// In BinnedDataset
is_categorical: Vec<bool>,  // Per-feature flag
```

Set during binning based on input feature type.

## Files

| Path | Contents |
| ---- | -------- |
| `training/gbdt/categorical.rs` | `CatBitset`, bitset operations |
| `training/gbdt/split/types.rs` | `SplitType`, `SplitInfo` |
| `training/gbdt/split/find.rs` | `GreedySplitter`, strategy dispatch |
| `repr/gbdt/categories.rs` | Tree category storage |

## Design Decisions

**DD-1: Sorted partition for high cardinality.** One-hot becomes expensive
above ~10 categories. Sorted partition is O(K log K) regardless of K and
often finds better partitions than brute-force.

**DD-2: Higher gradient → right.** Convention matches XGBoost. Sorted
partition naturally produces right-going sets when sorting ascending by
gradient ratio.

**DD-3: Categories in bitset go LEFT.** `left_cats` contains categories
that route samples left. Matches how numerical splits use "≤ threshold → left".

**DD-4: Max categories ~1000.** Sorted partition scratch buffer limits
practical cardinality. Beyond 1000, consider hashing or embedding.

**DD-5: Missing separate from categories.** Missing values are handled by
`default_left`, not as a special category bin. Keeps histogram logic simple.

## Unseen Categories

At prediction time, categories not seen during training are treated as missing:
- Follow `default_left` direction at categorical splits
- No error raised—graceful fallback

## High Cardinality Limits

**DD-4** limits practical cardinality to ~1000:
- CatBitset memory: 64 bytes inline + 8 bytes per 64 categories above 64
- Sorted partition scratch: ~16 bytes per category
- Beyond 1000 categories: consider target encoding, embedding, or hashing

If exceeded, training continues but may be slow and memory-intensive.

## Feature Marking

Mark features as categorical during Dataset construction:

```rust
// Via builder
let dataset = Dataset::builder()
    .add_feature("age", age_values.view())  // numeric by default
    .add_categorical("color", color_values.view())  // categorical
    .build()?;

// Via schema
let mut schema = DatasetSchema::default();
schema.set_feature_type(2, FeatureType::Categorical);
```

## Testing Strategy

| Category | Tests |
| -------- | ----- |
| One-hot splits | Each category as singleton works correctly |
| Sorted partition | Optimal partition found for known distributions |
| Bitset operations | Insert, contains, iterate correctness |
| Unseen categories | Prediction handles gracefully |
| High cardinality | 500+ categories don't crash |
