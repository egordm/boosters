# RFC-0019: Feature Value Iteration for Linear Models, SHAP, and Linear Trees

**Status**: Draft  
**Created**: 2025-12-30  
**Updated**: 2025-12-30  
**Author**: Team  
**Related**: RFC-0018 (Raw Feature Storage)

## Summary

Several components need efficient per-feature iteration over raw values:

- **GBLinear** (training and prediction): iterate all samples for each feature
- **Linear SHAP**: same pattern as GBLinear prediction (currently uses `FeaturesView`)
- **Linear tree fitting**: iterate a *subset* of samples (leaf rows) for each feature
- **Tree SHAP**: per-sample access across features—migrate to `SampleBlocks` for consistency

### Access Patterns

| Component           | Pattern                        | Current Implementation                  | New Pattern                |
| ------------------- | ------------------------------ | --------------------------------------- | -------------------------- |
| GBLinear training   | all samples, per feature       | `FeaturesView::feature(f)`              | `for_each_feature_value()` |
| GBLinear prediction | all samples, per feature       | `FeaturesView::feature(f)`              | `for_each_feature_value()` |
| Linear SHAP         | all samples, per feature       | `FeaturesView::feature(f)[i]`           | `for_each_feature_value()` |
| Linear tree fitting | subset of samples, per feature | `DataAccessor::sample(row).feature(f)`  | `gather_feature_values()`  |
| Tree SHAP           | per-sample, all features       | `FeaturesView::get(sample, feat)`       | `SampleBlocks`             |
| GBDT prediction     | per-sample batches             | `SampleBlocks`                          | (already efficient)        |

### Key Design Decisions

**1. `for_each_feature_value()` for full iteration** (GBLinear, Linear SHAP):

```rust
dataset.for_each_feature_value(feat_idx, |sample_idx, value| {
    // closure inlined directly into slice iteration
});
```

**2. `gather_feature_values()` for filtered iteration** (Linear tree fitting):

```rust
dataset.gather_feature_values(feat_idx, &sample_indices, &mut buffer);
// buffer now contains values for only the specified samples
```

**3. `SampleBlocks` for row-major access** (Tree SHAP):

```rust
for block in SampleBlocks::new(dataset, sample_indices) {
    for sample in block.iter() {
        let value = sample.feature(feat_idx);
    }
}
```

**Why `for_each` is zero-cost for dense features:**

- The storage type is matched **once** at the start of the call
- For dense features, the loop compiles to direct slice iteration
- The closure is inlined into the tight loop

**Why an enum iterator is NOT zero-cost:**

- `FeatureValueIter` must branch on every `.next()` call
- Even though the branch is predictable, it adds overhead (~5-10% in microbenchmarks)
- The iterator is still provided for ergonomics but documented as having overhead

**Key principle**: No legacy compatibility layer. We delete the old code and update all callers to work directly with `BinnedDataset`.

## Motivation

### GBLinear Access Pattern

```rust
// Training (updater.rs)
let feature_values = data.feature(feature);  // need: iter over (idx, value)
for (row, &value) in feature_values.iter().enumerate() {
    sum_grad += grad_hess[row].grad * value;
    sum_hess += grad_hess[row].hess * value * value;
}

// Prediction (model.rs)
for feat_idx in 0..n_features {
    let feature_values = features.feature(feat_idx);  // need: iter over (idx, value)
    for (sample_idx, &value) in feature_values.iter().enumerate() {
        output[[group, sample_idx]] += value * self.weight(feat_idx, group);
    }
}
```

**Problem**: We cannot return `&[f32]` for bundled features (EFB). If a feature is bundled, its raw values are encoded in the bundle and must be extracted per-sample. Returning a slice would require materializing the entire feature—defeating the purpose of sparse storage.

**Solution**: An iterator that yields `(sample_idx, f32)` pairs:

- For contiguous features: zero-cost wrapper over slice iter
- For bundled features: computes values on-the-fly from bundle encoding
- For sparse features: iterates only non-zero samples

### Linear Tree Fitting Access Pattern

```rust
// Current (per-sample access via DataAccessor)
for &row in leaf_samples {
    let sample = data.sample(row as usize);
    for &feat in features {
        x_buffer.push(sample.feature(feat));  // poor cache locality
    }
}
```

**Problem**: Per-sample access has poor cache locality when iterating over many features. The `DataAccessor` trait forces row-major access, but linear fitting naturally wants column-major (per-feature) access for building the design matrix.

**Key insight**: `leaf_samples` indices are **sorted** due to stable partitioning in tree growing. This enables efficient gather operations:

- **Dense features**: Trivial indexed gather `buffer[i] = slice[indices[i]]`
- **Categorical features**: Not used by linear trees (they only fit numeric features)
- **Sparse features**: Merge-join algorithm since both sparse indices and sample indices are sorted

**Solution**: A `gather_feature_values()` that fills a buffer with values for specified sample indices.

### Tree SHAP Access Pattern

```rust
// Current (per-sample, per-feature random access)
for sample_idx in 0..n_samples {
    for node in tree.nodes() {
        let value = features.get(sample_idx, node.feature);  // random access
        // ... tree traversal
    }
}
```

**Problem**: Random access pattern with poor cache locality. Each `get(sample_idx, feature)` may touch different cache lines.

**Solution**: Migrate to `SampleBlocks` (already used by GBDT prediction). This batches samples into cache-friendly blocks and provides row-major access within each block.

## Design

### Primary Pattern: `for_each_feature_value`

The `for_each` pattern is the **recommended approach** because it's truly zero-cost for dense features:

```rust
impl BinnedDataset {
    /// Apply a function to each (sample_idx, raw_value) pair for a feature.
    ///
    /// This is the recommended pattern for iterating over feature values.
    /// The storage type is matched ONCE, then we iterate directly on the
    /// underlying slice—no per-iteration branching.
    ///
    /// # Performance
    /// - Dense: Equivalent to `for (i, &v) in slice.iter().enumerate()`
    /// - Sparse: Iterates only stored (non-zero) values
    /// - Bundled: Extracts values from bundle encoding on-the-fly
    #[inline]
    pub fn for_each_feature_value<F>(&self, feature: usize, mut f: F)
    where
        F: FnMut(usize, f32),
    {
        let info = &self.features[feature];
        
        match info.location {
            FeatureLocation::Direct { group_idx, idx_in_group } => {
                let group = &self.groups[group_idx as usize];
                if let Some(slice) = group.raw_slice(idx_in_group as usize) {
                    // Dense: direct iteration, fully inlined
                    for (idx, &val) in slice.iter().enumerate() {
                        f(idx, val);
                    }
                } else if group.is_sparse() {
                    // Sparse: iterate stored values
                    group.for_each_sparse_value(idx_in_group as usize, |idx, val| f(idx, val));
                } else {
                    panic!("Feature {} is categorical", feature)
                }
            }
            FeatureLocation::Bundled { bundle_group_idx, position_in_bundle } => {
                // Bundled: extract from bundle
                let bundle = &self.groups[bundle_group_idx as usize];
                bundle.for_each_unbundled_value(position_in_bundle, |idx, val| f(idx, val));
            }
            FeatureLocation::Skipped => {
                panic!("Feature {} was skipped", feature)
            }
        }
    }
}
```

### Secondary: `FeatureValueIter` Enum (Has Overhead)

For cases where an `Iterator` is needed (e.g., chaining with other iterators, early return), we provide an enum iterator. **Note: This has ~5-10% overhead for dense features** due to branching on every `.next()` call.

```rust
/// Iterator yielding (sample_idx, raw_value) for a feature.
/// 
/// This abstracts over different storage strategies:
/// - Contiguous: Direct slice iteration (zero-cost)
/// - Bundled: Extracts values from EFB encoding
/// - Sparse: Yields only non-zero samples
pub enum FeatureValueIter<'a> {
    /// Contiguous storage - wraps slice::iter().enumerate()
    Dense(std::iter::Enumerate<std::slice::Iter<'a, f32>>),
    /// Bundled feature - extracts from bundle per sample
    Bundled(BundledFeatureIter<'a>),
    /// Sparse numeric feature - yields only stored samples
    Sparse(SparseFeatureIter<'a>),
}

impl<'a> Iterator for FeatureValueIter<'a> {
    type Item = (usize, f32);
    
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Dense(iter) => iter.next().map(|(idx, &val)| (idx, val)),
            Self::Bundled(iter) => iter.next(),
            Self::Sparse(iter) => iter.next(),
        }
    }
    
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Dense(iter) => iter.size_hint(),
            Self::Bundled(iter) => iter.size_hint(),
            Self::Sparse(iter) => iter.size_hint(),
        }
    }
}
```

### Gather Pattern: `gather_feature_values` (Linear Tree Fitting)

Linear tree fitting needs to iterate a **subset** of samples (leaf rows) for each feature. Currently it uses the `DataAccessor` trait with per-sample access (`sample(row).feature(feat)`), which has poor cache locality for per-feature operations.

We introduce a `gather_feature_values` pattern that fills a buffer with values for specified sample indices:

```rust
impl BinnedDataset {
    /// Gather raw values for a feature at specified sample indices into a buffer.
    ///
    /// This is the recommended pattern for linear tree fitting where we need
    /// values for a subset of samples (e.g., samples that landed in a leaf).
    ///
    /// # Arguments
    /// - `feature`: The feature index
    /// - `sample_indices`: Sorted slice of sample indices to gather
    /// - `buffer`: Output buffer, must have length >= sample_indices.len()
    ///
    /// # Performance
    /// - Dense: Simple indexed gather, O(k) where k = sample_indices.len()
    /// - Sparse: Merge-join since both indices and sparse storage are sorted
    /// - Bundled: Extract from bundle at each index
    ///
    /// # Note
    /// Linear trees only use numeric features, never categorical.
    #[inline]
    pub fn gather_feature_values(
        &self,
        feature: usize,
        sample_indices: &[u32],  // sorted due to stable partitioning
        buffer: &mut [f32],
    ) {
        debug_assert!(buffer.len() >= sample_indices.len());
        let info = &self.features[feature];
        
        match info.location {
            FeatureLocation::Direct { group_idx, idx_in_group } => {
                let group = &self.groups[group_idx as usize];
                if let Some(slice) = group.raw_slice(idx_in_group as usize) {
                    // Dense: trivial indexed gather
                    for (out_idx, &sample_idx) in sample_indices.iter().enumerate() {
                        buffer[out_idx] = slice[sample_idx as usize];
                    }
                } else if group.is_sparse() {
                    // Sparse: merge-join (both are sorted)
                    group.gather_sparse_values(idx_in_group as usize, sample_indices, buffer);
                } else {
                    panic!("Feature {} is categorical, linear trees don't use categorical", feature)
                }
            }
            FeatureLocation::Bundled { bundle_group_idx, position_in_bundle } => {
                // Bundled: extract at each index
                let bundle = &self.groups[bundle_group_idx as usize];
                bundle.gather_unbundled_values(position_in_bundle, sample_indices, buffer);
            }
            FeatureLocation::Skipped => {
                // Skipped features have constant value (usually 0)
                buffer[..sample_indices.len()].fill(0.0);
            }
        }
    }
    
    /// Similar to gather but with a callback for each (local_idx, value) pair.
    /// 
    /// Useful when you need to process values immediately without allocating.
    #[inline]
    pub fn for_each_gathered_value<F>(
        &self,
        feature: usize,
        sample_indices: &[u32],
        mut f: F,
    )
    where
        F: FnMut(usize, f32),  // (index into sample_indices, value)
    {
        let info = &self.features[feature];
        
        match info.location {
            FeatureLocation::Direct { group_idx, idx_in_group } => {
                let group = &self.groups[group_idx as usize];
                if let Some(slice) = group.raw_slice(idx_in_group as usize) {
                    // Dense: direct indexed access
                    for (out_idx, &sample_idx) in sample_indices.iter().enumerate() {
                        f(out_idx, slice[sample_idx as usize]);
                    }
                } else if group.is_sparse() {
                    // Sparse: merge-join iteration
                    group.for_each_gathered_sparse(idx_in_group as usize, sample_indices, |idx, val| f(idx, val));
                } else {
                    panic!("Feature {} is categorical", feature)
                }
            }
            FeatureLocation::Bundled { bundle_group_idx, position_in_bundle } => {
                let bundle = &self.groups[bundle_group_idx as usize];
                bundle.for_each_gathered_unbundled(position_in_bundle, sample_indices, |idx, val| f(idx, val));
            }
            FeatureLocation::Skipped => {
                for out_idx in 0..sample_indices.len() {
                    f(out_idx, 0.0);
                }
            }
        }
    }
}
```

#### Sparse Gather: Merge-Join Algorithm

For sparse features, both the sparse storage indices and the requested `sample_indices` are sorted. This enables an efficient merge-join:

```rust
impl FeatureGroup {
    fn gather_sparse_values(&self, feature_in_group: usize, sample_indices: &[u32], buffer: &mut [f32]) {
        // Initialize buffer with zeros (sparse default)
        buffer[..sample_indices.len()].fill(0.0);
        
        let (sparse_indices, sparse_values) = self.sparse_data(feature_in_group);
        
        // Merge-join: both are sorted
        let mut sparse_pos = 0;
        for (out_idx, &sample_idx) in sample_indices.iter().enumerate() {
            // Advance sparse_pos until we find sample_idx or pass it
            while sparse_pos < sparse_indices.len() && sparse_indices[sparse_pos] < sample_idx {
                sparse_pos += 1;
            }
            if sparse_pos < sparse_indices.len() && sparse_indices[sparse_pos] == sample_idx {
                buffer[out_idx] = sparse_values[sparse_pos];
            }
            // else: remains 0.0 (sparse default)
        }
    }
}
```

### Linear Tree Fitting Updates

```rust
// Before (uses DataAccessor per-sample)
fn fit_linear_model(
    data: &impl DataAccessor,
    leaf_samples: &[u32],
    features: &[usize],
) -> LinearLeafModel {
    let mut x_buffer = Vec::with_capacity(leaf_samples.len() * features.len());
    
    for &row in leaf_samples {
        let sample = data.sample(row as usize);
        for &feat in features {
            x_buffer.push(sample.feature(feat));
        }
    }
    // ... fit linear model
}

// After (uses gather_feature_values per-feature)
fn fit_linear_model(
    dataset: &BinnedDataset,
    leaf_samples: &[u32],  // sorted due to stable partitioning
    features: &[usize],
    feature_buffer: &mut [f32],  // reusable buffer, size = leaf_samples.len()
) -> LinearLeafModel {
    let n_samples = leaf_samples.len();
    let mut x_matrix = Vec::with_capacity(n_samples * features.len());
    
    for &feat_idx in features {
        dataset.gather_feature_values(feat_idx, leaf_samples, feature_buffer);
        x_matrix.extend_from_slice(&feature_buffer[..n_samples]);
    }
    // ... fit linear model (x_matrix is now column-major)
}
```

### Tree SHAP: Migrate to SampleBlocks

Tree SHAP currently uses per-sample access (`features.get(sample_idx, feature)`). For consistency with GBDT prediction and better cache locality, migrate to `SampleBlocks`:

```rust
// Before (per-sample access)
fn explain_sample(
    &self,
    features: &FeaturesView,
    sample_idx: usize,
    shap_values: &mut [f64],
) {
    for node in tree.nodes() {
        let feature_value = features.get(sample_idx, node.feature);
        // ... tree traversal logic
    }
}

// After (using SampleBlocks for batched access)
fn explain_batch(
    &self,
    blocks: &SampleBlocks,
    batch_start: usize,
    batch_size: usize,
    shap_values: &mut Array2<f64>,
) {
    for local_idx in 0..batch_size {
        let sample = blocks.sample(batch_start + local_idx);
        for node in tree.nodes() {
            let feature_value = sample.feature(node.feature);
            // ... tree traversal logic
        }
    }
}
```

This aligns Tree SHAP with GBDT prediction, which already uses `SampleBlocks` for efficient row-major access.

Add a single method to `BinnedDataset`:

```rust
impl BinnedDataset {
    /// Iterate over raw values for a feature.
    ///
    /// Returns an iterator yielding (sample_idx, raw_value) pairs.
    /// Panics if the feature is categorical (use binned access for those).
    ///
    /// # Feature Types
    /// - Dense numeric: yields all n_samples values in order
    /// - Sparse numeric: yields only non-zero samples  
    /// - Bundled (EFB): extracts and yields values from bundle encoding
    #[inline]
    pub fn feature_values(&self, feature: usize) -> FeatureValueIter<'_> {
        let info = &self.features[feature];
        
        match info.location {
            FeatureLocation::Direct { group_idx, idx_in_group } => {
                let group = &self.groups[group_idx as usize];
                if let Some(slice) = group.raw_slice(idx_in_group as usize) {
                    // Dense: direct slice iteration
                    FeatureValueIter::Dense(slice.iter().enumerate())
                } else if group.is_sparse() {
                    // Sparse: create sparse iterator
                    FeatureValueIter::Sparse(SparseFeatureIter::new(group, idx_in_group))
                } else {
                    panic!("Feature {} is categorical, use binned access", feature)
                }
            }
            FeatureLocation::Bundled { bundle_group_idx, position_in_bundle } => {
                // Bundled: extract from bundle encoding
                let bundle = &self.groups[bundle_group_idx as usize];
                FeatureValueIter::Bundled(BundledFeatureIter::new(
                    bundle, 
                    position_in_bundle,
                    self.n_samples,
                ))
            }
            FeatureLocation::Skipped => {
                panic!("Feature {} was skipped (trivial)", feature)
            }
        }
    }
    
    /// Iterate over all features that have raw values, yielding feature iterators.
    ///
    /// Skips categorical features. For each numeric feature, yields
    /// (feature_idx, FeatureValueIter).
    pub fn iter_feature_values(&self) 
        -> impl Iterator<Item = (usize, FeatureValueIter<'_>)> + '_ 
    {
        (0..self.n_features()).filter_map(move |idx| {
            // Skip categorical features
            if self.features[idx].is_categorical() {
                return None;
            }
            // Skip skipped features
            if self.features[idx].location.is_skipped() {
                return None;
            }
            Some((idx, self.feature_values(idx)))
        })
    }
}
```

### GBLinear Updates

#### LinearModel Prediction

```rust
// Before (uses FeaturesView)
pub fn predict_into(&self, features: FeaturesView<'_>, mut output: ArrayViewMut2<'_, f32>) {
    for feat_idx in 0..n_features {
        let feature_values = features.feature(feat_idx);
        for (sample_idx, &value) in feature_values.iter().enumerate() {
            for group in 0..n_groups {
                output[[group, sample_idx]] += value * self.weight(feat_idx, group);
            }
        }
    }
}

// After (uses BinnedDataset directly)
pub fn predict_into(&self, dataset: &BinnedDataset, mut output: ArrayViewMut2<'_, f32>) {
    for feat_idx in 0..n_features {
        for (sample_idx, value) in dataset.feature_values(feat_idx) {
            for group in 0..n_groups {
                output[[group, sample_idx]] += value * self.weight(feat_idx, group);
            }
        }
    }
}

// Or with for_each pattern:
pub fn predict_into(&self, dataset: &BinnedDataset, mut output: ArrayViewMut2<'_, f32>) {
    for feat_idx in 0..n_features {
        dataset.for_each_feature_value(feat_idx, |sample_idx, value| {
            for group in 0..n_groups {
                output[[group, sample_idx]] += value * self.weight(feat_idx, group);
            }
        });
    }
}
```

#### Updater

```rust
// Before
fn compute_weight_update(
    model: &LinearModel,
    data: &FeaturesView<'_>,
    buffer: &Gradients,
    feature: usize,
    output: usize,
    config: &UpdateConfig,
) -> f32 {
    let feature_values = data.feature(feature);
    for (row, &value) in feature_values.iter().enumerate() {
        sum_grad += grad_hess[row].grad * value;
        sum_hess += grad_hess[row].hess * value * value;
    }
    // ...
}

// After
fn compute_weight_update(
    model: &LinearModel,
    data: &BinnedDataset,
    buffer: &Gradients,
    feature: usize,
    output: usize,
    config: &UpdateConfig,
) -> f32 {
    for (row, value) in data.feature_values(feature) {
        sum_grad += grad_hess[row].grad * value;
        sum_hess += grad_hess[row].hess * value * value;
    }
    // ...
}
```

## Performance Analysis

### `for_each` Pattern (Recommended)

For the `for_each` pattern, the storage type is matched **once** at the start of the call. After that, we iterate directly on the underlying data:

**Dense features:**

```rust
// This is what for_each compiles to for dense features:
for (idx, &val) in slice.iter().enumerate() {
    f(idx, val);  // closure is inlined
}
```

**Expected overhead**: 0% (identical to direct slice iteration)

### `FeatureValueIter` Enum (Has Overhead)

For the enum iterator, every `.next()` call branches on the variant:

```rust
fn next(&mut self) -> Option<Self::Item> {
    match self {  // <-- branch on EVERY iteration
        Self::Dense(iter) => iter.next().map(|(idx, &val)| (idx, val)),
        Self::Bundled(iter) => iter.next(),
        Self::Sparse(iter) => iter.next(),
    }
}
```

**Expected overhead**: ~5-10% for dense features (predictable branch, but still present)

**When to use the iterator anyway:**

- When you need to chain with other iterators (`.zip()`, `.take()`, etc.)
- When you need early return (can't easily do with `for_each`)
- When the overhead is acceptable for your use case

### Bundled Features (EFB)

For bundled features, we extract values from the bundle per sample. This is inherently more expensive than direct slice access, but:

1. We're not materializing a full slice upfront
2. The extraction is done lazily during iteration
3. GBLinear typically has few bundled features (EFB bundles correlated features)

### Sparse Features

Sparse iteration yields only non-zero samples, which is the correct behavior for linear models (zero values don't contribute to the sum).

## Migration Plan

**No legacy compatibility**. The migration is:

### Phase 1: Core API

1. Add `for_each_feature_value()` to `BinnedDataset` (full iteration pattern)
2. Add `gather_feature_values()` and `for_each_gathered_value()` (filtered iteration pattern)
3. Add `FeatureValueIter` enum and `feature_values()` (secondary, for ergonomics)

### Phase 2: GBLinear Migration

1. Update `LinearModel::predict_into()` to take `&BinnedDataset`
2. Update `Updater` methods to take `&BinnedDataset`
3. Update `GBLinearTrainer::train()` to pass `&BinnedDataset`

### Phase 3: Linear SHAP Migration

1. Update `LinearExplainer` to use `for_each_feature_value()`
   - Currently uses `features.feature(f)[i]` pattern
   - Same migration pattern as GBLinear prediction

### Phase 4: Linear Tree Fitting Migration

1. Update `LeafLinearTrainer` to use `gather_feature_values()`
   - Currently uses `DataAccessor::sample(row).feature(feat)`
   - Switch to per-feature gather into reusable buffer
   - Note: sample indices are sorted due to stable partitioning

### Phase 5: Tree SHAP Migration

1. Update `TreeExplainer` to use `SampleBlocks`
   - Currently uses `FeaturesView::get(sample_idx, feature)`
   - Migrate to batched `SampleBlocks` access for consistency with GBDT prediction
   - Better cache locality for row-major access pattern

### Phase 6: Cleanup

1. Update Python bindings
2. Delete `FeaturesView`, `DataAccessor` trait, and old `Dataset` type

The library will not compile during phases 2-6. That's acceptable—we fix all callers and then it compiles again.

## Open Questions

1. **~~Iterator vs for_each~~**: **Resolved.** `for_each` is the primary pattern (zero-cost for dense). Iterator is provided for ergonomics but documented as having ~5-10% overhead.

2. **Sparse handling in GBLinear**: When a sparse feature yields only non-zero samples, the gradient update skips zero samples. This is mathematically correct (zero × weight = 0), but we need to ensure the outer loop handles this correctly.

3. **Bundle extraction cost**: How expensive is extracting a value from a bundle? If significant, consider caching or batch extraction.

4. **~~Linear tree fitting~~**: **Resolved.** Use `gather_feature_values()` with sorted sample indices. Merge-join algorithm for sparse features leverages the fact that indices are sorted due to stable partitioning.

5. **~~Tree SHAP~~**: **Resolved.** Migrate to `SampleBlocks` for consistency with GBDT prediction and better cache locality.

## Success Criteria

- [ ] `for_each_feature_value()` implemented on `BinnedDataset`
- [ ] `gather_feature_values()` implemented on `BinnedDataset`
- [ ] `for_each_gathered_value()` implemented on `BinnedDataset`
- [ ] `FeatureValueIter` enum implemented (secondary API)
- [ ] GBLinear training works with `BinnedDataset` (bundled features included)
- [ ] GBLinear prediction works with `BinnedDataset`
- [ ] Linear SHAP works with `BinnedDataset`
- [ ] Linear tree fitting works with `gather_feature_values()`
- [ ] Tree SHAP works with `SampleBlocks`
- [ ] Old `FeaturesView`, `DataAccessor`, and `Dataset` types deleted
- [ ] No O(n×m) allocations in the path
- [ ] Benchmark shows `for_each` has 0% overhead for dense features
- [ ] Benchmark shows iterator has <10% overhead for dense features
- [ ] Benchmark shows gather is efficient for sorted indices
