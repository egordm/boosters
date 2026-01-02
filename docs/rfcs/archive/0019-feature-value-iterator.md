# RFC-0019: Feature Value Iteration for Linear Models, SHAP, and Linear Trees

**Status**: Implemented  
**Created**: 2025-12-30  
**Updated**: 2026-01-02  
**Author**: Team  
**Related**: RFC-0021 (Dataset/BinnedDataset Separation)

## Summary

Several components need efficient per-feature iteration over raw values:

- **GBLinear** (training and prediction): iterate all samples for each feature
- **Linear SHAP**: same pattern as GBLinear prediction
- **Linear tree fitting**: iterate a *subset* of samples (leaf rows) for each feature
- **Tree SHAP**: per-sample access across features—uses `buffer_samples()`
- **GBDT prediction**: per-sample batches—uses `buffer_samples()`

### Key Design Decision: Methods Live on Dataset

Per RFC-0021, `Dataset` contains raw feature values (dense or sparse), while `BinnedDataset`
contains only bins and is internal to the library. All raw value iteration methods therefore
belong on `Dataset`, not `BinnedDataset`.

**Why Dataset, not BinnedDataset?**

- `Dataset` owns raw values; `BinnedDataset` only has bins
- `Dataset` is the public API; users pass it to models
- Prediction, SHAP, GBLinear, linear trees all work with raw values
- Only histogram building needs bins (which uses `BinnedDataset` internally)

### Access Patterns

| Component | Pattern | Current Implementation | New Pattern |
| --- | --- | --- | --- |
| GBLinear training | all samples, per feature | `FeaturesView::feature(f)` | `Dataset::for_each_feature_value()` |
| GBLinear prediction | all samples, per feature | `FeaturesView::feature(f)` | `Dataset::for_each_feature_value()` |
| Linear SHAP | all samples, per feature | `FeaturesView::feature(f)[i]` | `Dataset::for_each_feature_value()` |
| Linear tree fitting | subset of samples, per feature | `DataAccessor::sample(row).feature(f)` | `Dataset::gather_feature_values()` |
| Tree SHAP | per-sample, all features | `FeaturesView::get(sample, feat)` | `Dataset::buffer_samples()` + caller loop |
| GBDT prediction | per-sample batches | `SampleBlocks` (on BinnedDataset) | `Dataset::buffer_samples()` + caller loop |

### Key Design Decisions

**1. `for_each_feature_value()` on Dataset** (GBLinear, Linear SHAP):

```rust
dataset.for_each_feature_value(feat_idx, |sample_idx, value| {
    // closure inlined directly into slice iteration
});
```

**2. `gather_feature_values()` on Dataset** (Linear tree fitting):

```rust
dataset.gather_feature_values(feat_idx, &sample_indices, &mut buffer);
// buffer now contains values for only the specified samples
```

**3. `buffer_samples()` on Dataset** (Tree SHAP, GBDT prediction):

Instead of a dedicated `SampleBlocks` iterator type, callers manage their own sample-major
buffers and use `Parallelism::maybe_par_for_each_init` for parallel processing with
per-thread buffer reuse:

```rust
// Caller allocates buffer once per thread via for_each_init
let n_blocks = (n_samples + block_size - 1) / block_size;

parallelism.maybe_par_for_each_init(
    0..n_blocks,
    // Init: create buffer once per thread
    || Array2::<f32>::zeros((block_size, n_features)),
    // Process: fill buffer and use it
    |buffer, block_idx| {
        let start = block_idx * block_size;
        let filled = dataset.buffer_samples(&mut buffer.view_mut(), start);
        
        // Process samples 0..filled in the buffer
        for sample_idx in 0..filled {
            let sample = buffer.row(sample_idx);
            // ... prediction or SHAP computation
        }
    },
);
```

**Why caller-managed buffers instead of SampleBlocks:**

- Simpler design—no iterator wrapper type with lifetime management
- More flexible—caller can decide buffer size and parallelism strategy
- Less unsafe code—no raw pointer tricks for parallel output access
- Rayon's `for_each_init` handles per-thread buffer reuse naturally

**Why `for_each` is zero-cost for dense features:**

- The storage type is matched **once** at the start of the call
- For dense features, the loop compiles to direct slice iteration
- The closure is inlined into the tight loop

**Why an enum iterator is NOT zero-cost:**

- `FeatureValueIter` must branch on every `.next()` call
- Even though the branch is predictable, it adds overhead (~5-10% in microbenchmarks)
- The iterator is still provided for ergonomics but documented as having overhead

**Key principles:**

1. **All iteration methods on `Dataset`** - BinnedDataset only has bins, not raw values
2. **No legacy compatibility layer** - Delete old code, update all callers
3. **FeaturesView removed** - It assumed dense contiguous data; Dataset handles both dense and sparse

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

**Solution**: Use `buffer_samples()` with caller-managed buffers. This batches samples into cache-friendly blocks and provides row-major access within each block.

## Design

### Primary Pattern: `for_each_feature_value` on Dataset

The `for_each` pattern is the **recommended approach** because it's truly zero-cost for dense features:

```rust
impl Dataset {
    /// Apply a function to each (sample_idx, raw_value) pair for a feature.
    ///
    /// This is the recommended pattern for iterating over feature values.
    /// The storage type is matched ONCE, then we iterate directly on the
    /// underlying data—no per-iteration branching.
    ///
    /// # Performance
    /// - Dense: Equivalent to `for (i, &v) in slice.iter().enumerate()`
    /// - Sparse: Iterates only stored (non-default) values
    #[inline]
    pub fn for_each_feature_value<F>(&self, feature: usize, mut f: F)
    where
        F: FnMut(usize, f32),
    {
        match &self.features[feature] {
            Feature::Dense(values) => {
                // Dense: direct iteration, fully inlined
                for (idx, &val) in values.iter().enumerate() {
                    f(idx, val);
                }
            }
            Feature::Sparse { indices, values, default: _ } => {
                // Sparse: iterate only stored values
                for (&idx, &val) in indices.iter().zip(values.iter()) {
                    f(idx as usize, val);
                }
            }
        }
    }
    
    /// Iterate over all samples, filling in defaults for sparse features.
    ///
    /// Use this when you need ALL samples, including default values.
    #[inline]
    pub fn for_each_feature_value_dense<F>(&self, feature: usize, mut f: F)
    where
        F: FnMut(usize, f32),
    {
        match &self.features[feature] {
            Feature::Dense(values) => {
                for (idx, &val) in values.iter().enumerate() {
                    f(idx, val);
                }
            }
            Feature::Sparse { indices, values, default } => {
                // For sparse, we need to iterate all samples and fill gaps
                let mut sparse_pos = 0;
                for sample_idx in 0..self.n_samples {
                    if sparse_pos < indices.len() && indices[sparse_pos] == sample_idx as u32 {
                        f(sample_idx, values[sparse_pos]);
                        sparse_pos += 1;
                    } else {
                        f(sample_idx, *default);
                    }
                }
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
/// - Dense: Direct slice iteration (near zero-cost)
/// - Sparse: Yields only stored (non-default) samples
pub enum FeatureValueIter<'a> {
    /// Dense storage - wraps slice::iter().enumerate()
    Dense(std::iter::Enumerate<std::slice::Iter<'a, f32>>),
    /// Sparse storage - yields only stored samples
    Sparse(SparseFeatureIter<'a>),
}

impl<'a> Iterator for FeatureValueIter<'a> {
    type Item = (usize, f32);
    
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {  // <-- branch on EVERY iteration
            Self::Dense(iter) => iter.next().map(|(idx, &val)| (idx, val)),
            Self::Sparse(iter) => iter.next(),
        }
    }
    
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Dense(iter) => iter.size_hint(),
            Self::Sparse(iter) => iter.size_hint(),
        }
    }
}
```

**Note**: No `Bundled` variant. EFB bundling is a `BinnedDataset` concern for histogram building.
`Dataset` stores features as simple dense or sparse columns—no bundles.

### Gather Pattern: `gather_feature_values` on Dataset (Linear Tree Fitting)

Linear tree fitting needs to iterate a **subset** of samples (leaf rows) for each feature. Currently it uses the `DataAccessor` trait with per-sample access (`sample(row).feature(feat)`), which has poor cache locality for per-feature operations.

We introduce a `gather_feature_values` pattern on `Dataset` that fills a buffer with values for specified sample indices:

```rust
impl Dataset {
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
        
        match &self.features[feature] {
            Feature::Dense(values) => {
                // Dense: trivial indexed gather
                for (out_idx, &sample_idx) in sample_indices.iter().enumerate() {
                    buffer[out_idx] = values[sample_idx as usize];
                }
            }
            Feature::Sparse { indices, values, default } => {
                // Sparse: merge-join (both are sorted)
                Self::gather_sparse_values(indices, values, *default, sample_indices, buffer);
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
        match &self.features[feature] {
            Feature::Dense(values) => {
                for (out_idx, &sample_idx) in sample_indices.iter().enumerate() {
                    f(out_idx, values[sample_idx as usize]);
                }
            }
            Feature::Sparse { indices, values, default } => {
                // Sparse: merge-join iteration
                Self::for_each_gathered_sparse(indices, values, *default, sample_indices, |idx, val| f(idx, val));
            }
        }
    }
}
```

#### Sparse Gather: Merge-Join Algorithm

For sparse features, both the sparse storage indices and the requested `sample_indices` are sorted. This enables an efficient merge-join:

```rust
impl Dataset {
    fn gather_sparse_values(
        sparse_indices: &[u32],
        sparse_values: &[f32],
        default: f32,
        sample_indices: &[u32],
        buffer: &mut [f32],
    ) {
        // Initialize buffer with default (sparse default value)
        buffer[..sample_indices.len()].fill(default);
        
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
            // else: remains default
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

// After (uses Dataset.gather_feature_values per-feature)
fn fit_linear_model(
    dataset: &Dataset,
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

### Tree SHAP and GBDT Prediction: buffer_samples on Dataset

Tree SHAP and GBDT prediction both use row-major access patterns. Instead of a dedicated
`SampleBlocks` type, we use `Dataset::buffer_samples()` with caller-managed buffers:

```rust
// Before (SampleBlocks on BinnedDataset with FeaturesView)
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

// After (buffer_samples with caller-managed buffers)
fn explain_batch(
    &self,
    dataset: &Dataset,
    shap_values: &mut Array2<f64>,
    parallelism: Parallelism,
) {
    let n_samples = dataset.n_samples();
    let n_features = dataset.n_features();
    let block_size = 64;
    let n_blocks = (n_samples + block_size - 1) / block_size;

    parallelism.maybe_par_for_each_init(
        0..n_blocks,
        || Array2::<f32>::zeros((block_size, n_features)),
        |buffer, block_idx| {
            let start = block_idx * block_size;
            let filled = dataset.buffer_samples(&mut buffer.view_mut(), start);
            
            for local_idx in 0..filled {
                let sample = buffer.row(local_idx);
                let sample_idx = start + local_idx;
                for node in tree.nodes() {
                    let feature_value = sample[node.feature];
                    // ... tree traversal logic
                }
            }
        },
    );
}
```

**Key change**: No `SampleBlocks` type. Callers manage their own buffers and use
`Parallelism::maybe_par_for_each_init` for per-thread buffer reuse. This is simpler
and more flexible.

### Convenience Method: `feature_values()` on Dataset

For ergonomics, we add a method that returns an iterator:

```rust
impl Dataset {
    /// Iterate over raw values for a feature.
    ///
    /// Returns an iterator yielding (sample_idx, raw_value) pairs.
    ///
    /// # Feature Types
    /// - Dense: yields all n_samples values in order
    /// - Sparse: yields only stored (non-default) samples
    #[inline]
    pub fn feature_values(&self, feature: usize) -> FeatureValueIter<'_> {
        match &self.features[feature] {
            Feature::Dense(values) => {
                FeatureValueIter::Dense(values.iter().enumerate())
            }
            Feature::Sparse { indices, values, .. } => {
                FeatureValueIter::Sparse(SparseFeatureIter::new(indices, values))
            }
        }
    }
    
    /// Iterate over all features, yielding feature iterators.
    pub fn iter_feature_values(&self) 
        -> impl Iterator<Item = (usize, FeatureValueIter<'_>)> + '_ 
    {
        (0..self.n_features()).map(move |idx| {
            (idx, self.feature_values(idx))
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

// After (uses Dataset directly)
pub fn predict_into(&self, dataset: &Dataset, mut output: ArrayViewMut2<'_, f32>) {
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

// After (uses Dataset directly)
fn compute_weight_update(
    model: &LinearModel,
    data: &Dataset,
    buffer: &Gradients,
    feature: usize,
    output: usize,
    config: &UpdateConfig,
) -> f32 {
    data.for_each_feature_value(feature, |row, value| {
        sum_grad += grad_hess[row].grad * value;
        sum_hess += grad_hess[row].hess * value * value;
    });
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

### Sparse Features

Sparse iteration yields only stored (non-default) samples, which is the correct behavior for
linear models (default values contribute `default × weight` to the sum). For the common case
where default is 0, this is optimal.

## Migration Plan

**No legacy compatibility**. The migration is:

### Phase 1: Dataset Core API (with RFC-0021)

1. Implement `Dataset` type with `Feature::Dense` and `Feature::Sparse`
2. Add `for_each_feature_value()` to `Dataset` (full iteration pattern)
3. Add `for_each_feature_value_dense()` for cases needing all samples including defaults
4. Add `gather_feature_values()` and `for_each_gathered_value()` (filtered iteration)
5. Add `FeatureValueIter` enum and `feature_values()` (secondary, for ergonomics)
6. Add `buffer_samples()` for sample-major buffer filling

### Phase 2: GBLinear Migration

1. Update `LinearModel::predict_into()` to take `&Dataset`
2. Update `Updater` methods to take `&Dataset`
3. Update `GBLinearTrainer::train()` to pass `&Dataset`

### Phase 3: Linear SHAP Migration

1. Update `LinearExplainer` to use `Dataset::for_each_feature_value()`
   - Currently uses `features.feature(f)[i]` pattern
   - Same migration pattern as GBLinear prediction

### Phase 4: Linear Tree Fitting Migration

1. Update `LeafLinearTrainer` to use `Dataset::gather_feature_values()`
   - Currently uses `DataAccessor::sample(row).feature(feat)`
   - Switch to per-feature gather into reusable buffer
   - Note: sample indices are sorted due to stable partitioning

### Phase 5: Tree SHAP and GBDT Prediction Migration

1. Update `TreeExplainer` to use `buffer_samples()` on `Dataset`
2. Update GBDT prediction to use `buffer_samples()` on `Dataset`
   - Currently uses `SampleBlocks` on BinnedDataset
   - Migrate to caller-managed buffers with `buffer_samples()`

### Phase 6: Cleanup

1. Update Python bindings
2. Delete `FeaturesView` (assumed dense data—replaced by Dataset patterns)
3. Delete `DataAccessor` trait
4. Delete old `Dataset` type (if any remnants)

The library will not compile during phases 2-6. That's acceptable—we fix all callers and then it compiles again.

## Open Questions

1. **~~Iterator vs for_each~~**: **Resolved.** `for_each` is the primary pattern (zero-cost for dense). Iterator is provided for ergonomics but documented as having ~5-10% overhead.

2. **Sparse handling in GBLinear**: When a sparse feature yields only non-default samples, the gradient update skips default samples. For linear models this is mathematically correct when default=0 (zero × weight = 0). Need to verify this holds for all cases.

3. **~~Linear tree fitting~~**: **Resolved.** Use `gather_feature_values()` with sorted sample indices. Merge-join algorithm for sparse features leverages the fact that indices are sorted due to stable partitioning.

4. **~~Tree SHAP~~**: **Resolved.** Use `buffer_samples()` on Dataset with caller-managed buffers for consistency with GBDT prediction and better cache locality.

## Success Criteria

- [ ] `Dataset` type implemented with `Feature::Dense` and `Feature::Sparse`
- [ ] `for_each_feature_value()` implemented on `Dataset`
- [ ] `for_each_feature_value_dense()` implemented on `Dataset`
- [ ] `gather_feature_values()` implemented on `Dataset`
- [ ] `for_each_gathered_value()` implemented on `Dataset`
- [ ] `FeatureValueIter` enum implemented (secondary API)
- [ ] `buffer_samples()` implemented on Dataset
- [ ] GBLinear training works with `Dataset`
- [ ] GBLinear prediction works with `Dataset`
- [ ] Linear SHAP works with `Dataset`
- [ ] Linear tree fitting works with `Dataset::gather_feature_values()`
- [ ] Tree SHAP works with `buffer_samples()` on `Dataset`
- [ ] GBDT prediction works with `buffer_samples()` on `Dataset`
- [ ] Old `FeaturesView` deleted (assumed dense data)
- [ ] Old `DataAccessor` trait deleted
- [ ] No O(n×m) allocations in the path
- [ ] Benchmark shows `for_each` has 0% overhead for dense features
- [ ] Benchmark shows iterator has <10% overhead for dense features
- [ ] Benchmark shows gather is efficient for sorted indices
