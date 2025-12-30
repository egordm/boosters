# RFC-0019: Feature Value Iterator for GBLinear

**Status**: Draft  
**Created**: 2025-12-30  
**Updated**: 2025-12-30  
**Author**: Team  
**Related**: RFC-0018 (Raw Feature Storage)

## Summary

GBLinear needs to iterate over `(sample_idx, f32)` pairs per feature for both training and prediction. This RFC defines a zero-cost abstraction: an iterator that works uniformly whether the feature is stored contiguously (direct slice) or computed on-the-fly (bundled/sparse features).

**Key principle**: No legacy compatibility layer. We delete the old code and update GBLinear to work directly with `BinnedDataset`. The library may not compile during migration—that's fine.

## Motivation

GBLinear training and prediction have the same access pattern:

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

## Design

### Core Abstraction: `FeatureValueIter`

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

### BinnedDataset API

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

### Alternative: `for_each` Pattern

If the generic iterator return type causes issues, use a callback pattern:

```rust
impl BinnedDataset {
    /// Apply a function to each (sample_idx, raw_value) pair for a feature.
    ///
    /// This avoids the need for a generic return type while remaining zero-cost.
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

### Dense Features (Common Case)

For contiguous numeric features, the iterator is:
```rust
FeatureValueIter::Dense(slice.iter().enumerate())
```

The match on enum variant is a single branch prediction. The inner iteration is `slice.iter().enumerate()` which compiles to the same code as the original.

**Expected overhead**: ~0% (same as before).

### Bundled Features (EFB)

For bundled features, we extract values from the bundle per sample. This is inherently more expensive than a direct slice access, but:
1. We're not materializing a full slice upfront
2. The extraction is done lazily during iteration
3. GBLinear typically has few bundled features (EFB bundles correlated features)

### Sparse Features

Sparse iteration yields only non-zero samples, which is the correct behavior for linear models (zero values don't contribute to the sum).

## Migration Plan

**No legacy compatibility**. The migration is:

1. Add `FeatureValueIter` enum and `feature_values()` to `BinnedDataset`
2. Update `LinearModel::predict_into()` to take `&BinnedDataset`
3. Update `Updater` methods to take `&BinnedDataset`
4. Update `GBLinearTrainer::train()` to pass `&BinnedDataset`
5. Update Python bindings
6. Delete `FeaturesView` and old `Dataset` type

The library will not compile during steps 2-5. That's acceptable—we fix all callers and then it compiles again.

## Open Questions

1. **Iterator vs for_each**: The iterator approach is more idiomatic Rust. The for_each approach may inline better. Benchmark both.

2. **Sparse handling in GBLinear**: When a sparse feature yields only non-zero samples, the gradient update skips zero samples. This is mathematically correct (zero × weight = 0), but we need to ensure the outer loop handles this correctly.

3. **Bundle extraction cost**: How expensive is extracting a value from a bundle? If significant, consider caching or batch extraction.

## Success Criteria

- [ ] GBLinear training works with BinnedDataset (bundled features included)
- [ ] GBLinear prediction works with BinnedDataset
- [ ] Old `FeaturesView` and `Dataset` types deleted
- [ ] No O(n×m) allocations in the path
- [ ] Benchmark shows <5% overhead for dense features vs baseline
