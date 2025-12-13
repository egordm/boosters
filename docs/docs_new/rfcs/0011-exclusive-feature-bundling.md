```markdown
# RFC-0011: Exclusive Feature Bundling (EFB)

- **Status**: Draft
- **Created**: 2024-12-05
- **Updated**: 2024-12-05
- **Depends on**: RFC-0008 (Quantization), RFC-0009 (Histogram Building)
- **Scope**: Bundling sparse mutually-exclusive features to reduce histogram building cost

## Summary

Exclusive Feature Bundling (EFB) groups sparse features that rarely have non-zero values simultaneously into bundles. Each bundle is treated as a single "super-feature" during histogram building, reducing the effective feature count and improving cache efficiency. This is LightGBM's key innovation for sparse data.

## Motivation

Real-world datasets often have many sparse features:

| Dataset Type | Typical Sparsity | Features | Non-Zero Rate |
|--------------|------------------|----------|---------------|
| One-hot encoded | >95% | 100s-1000s | <5% |
| Text TF-IDF | >99% | 10000s | <1% |
| Click logs | >90% | 1000s | <10% |
| Genomics | >95% | 10000s | <5% |

Without EFB, histogram building iterates all features even though most values are zero/missing for any given sample. EFB bundles mutually-exclusive features together:

```text
Before: 1000 sparse features → 1000 histogram builds per node
After:  50 bundles          → 50 histogram builds per node (20× faster)
```

## Core Concept

### Mutual Exclusivity

Two features are **mutually exclusive** if they rarely both have non-zero values for the same sample:

```text
Feature A: [0, 1, 0, 0, 2, 0, 0]
Feature B: [3, 0, 0, 4, 0, 0, 0]
            ↑           ↑
         Conflict: both non-zero at same row

Conflict rate = conflicts / min(nnz_A, nnz_B) = 0 / 2 = 0%
→ These features can be bundled
```

Allowing a small conflict rate (e.g., 5%) enables more aggressive bundling with minimal accuracy loss.

### Bin Offset Encoding

Bundled features share a single bin array with offset encoding:

```text
Feature A bins: [0, 1, 2]     → 3 bins
Feature B bins: [0, 1, 2, 3]  → 4 bins

Bundle encoding:
  Feature A value → bin 0-2
  Feature B value → bin 3-6 (offset by A's bin count)
  Missing both    → special missing bin

bundled_bin = if A non-zero: A_bin
              else if B non-zero: A_n_bins + B_bin
              else: missing_bin
```

## Components

### FeatureBundle

```rust
/// A bundle of mutually-exclusive features.
pub struct FeatureBundle {
    /// Original feature indices in this bundle.
    pub feature_indices: Vec<u32>,
    /// Bin offset for each feature in the bundle.
    pub bin_offsets: Vec<u32>,  // [0, n_bins_f0, n_bins_f0+f1, ...]
    /// Total bins in the bundle.
    pub total_bins: u32,
}

impl FeatureBundle {
    /// Encode a value from one of the bundled features.
    pub fn encode(&self, local_feature_idx: usize, bin: u16) -> u16 {
        (self.bin_offsets[local_feature_idx] + bin as u32) as u16
    }
    
    /// Decode back to (local_feature_idx, original_bin).
    pub fn decode(&self, bundled_bin: u16) -> (usize, u16);
}
```

### FeatureBundler

Computes optimal bundles from data:

```rust
pub struct FeatureBundler {
    /// Maximum allowed conflict rate (default: 0.0 = exact exclusivity).
    pub max_conflict_rate: f32,
    /// Maximum features per bundle (default: no limit).
    pub max_bundle_size: usize,
}

impl FeatureBundler {
    /// Compute bundles from quantized data.
    pub fn compute_bundles(
        &self,
        quantized: &QuantizedMatrix,
        cuts: &BinCuts,
    ) -> Vec<FeatureBundle>;
}
```

### BundledMatrix

Quantized matrix with bundled features:

```rust
pub struct BundledMatrix {
    /// Bundled bin indices.
    data: QuantizedStorage,
    /// Bundle definitions.
    bundles: Vec<FeatureBundle>,
    /// Mapping: original_feature → (bundle_idx, local_idx).
    feature_to_bundle: Vec<(u32, u32)>,
    n_rows: usize,
}

impl BundledMatrix {
    /// Number of bundles (effective features for histogram building).
    pub fn n_bundles(&self) -> usize;
    
    /// Get bundled bin for (bundle, row).
    pub fn get_bin(&self, bundle: usize, row: usize) -> u16;
    
    /// Get bundle definition.
    pub fn bundle(&self, idx: usize) -> &FeatureBundle;
}
```

## Algorithms

### Bundle Discovery (Greedy Graph Coloring)

Model feature bundling as graph coloring where edges connect conflicting features:

```text
build_feature_graph(quantized, max_conflict_rate):
  n_features = quantized.n_features()
  
  // Compute non-zero masks per feature
  nnz_masks = []
  for f in 0..n_features:
    mask = BitSet::with_capacity(n_rows)
    for row in 0..n_rows:
      if quantized.get_bin(f, row) != missing_bin:
        mask.set(row)
    nnz_masks.push(mask)
  
  // Build conflict graph
  conflicts = vec![vec![]; n_features]
  for i in 0..n_features:
    for j in i+1..n_features:
      intersection = nnz_masks[i].intersection(&nnz_masks[j]).count()
      min_nnz = min(nnz_masks[i].count(), nnz_masks[j].count())
      
      if min_nnz > 0:
        conflict_rate = intersection as f32 / min_nnz as f32
        if conflict_rate > max_conflict_rate:
          conflicts[i].push(j)
          conflicts[j].push(i)
  
  return conflicts
```

### Greedy Bundling

```text
greedy_bundle(conflicts, n_features, max_bundle_size):
  bundles = []
  bundled = [false; n_features]
  
  // Sort features by conflict count (ascending = easier to bundle first)
  order = (0..n_features).sorted_by(|a, b| conflicts[a].len().cmp(&conflicts[b].len()))
  
  for f in order:
    if bundled[f]: continue
    
    // Try to add to existing bundle
    added = false
    for bundle in &mut bundles:
      if bundle.len() >= max_bundle_size: continue
      
      // Check if f conflicts with any feature in bundle
      has_conflict = bundle.iter().any(|&b| conflicts[f].contains(&b))
      if !has_conflict:
        bundle.push(f)
        bundled[f] = true
        added = true
        break
    
    // Create new bundle if needed
    if !added:
      bundles.push(vec![f])
      bundled[f] = true
  
  return bundles
```

### Matrix Bundling

```text
bundle_matrix(quantized, cuts, bundles):
  n_rows = quantized.n_rows()
  
  // Compute bin offsets for each bundle
  feature_bundles = []
  for bundle_features in bundles:
    offsets = [0]
    for f in bundle_features:
      offsets.push(offsets.last() + cuts.n_bins(f))
    
    feature_bundles.push(FeatureBundle {
      feature_indices: bundle_features,
      bin_offsets: offsets[..offsets.len()-1],
      total_bins: offsets.last(),
    })
  
  // Encode bundled values
  data = []
  for bundle_idx, bundle in enumerate(feature_bundles):
    for row in 0..n_rows:
      bundled_bin = missing_bin
      
      for (local_idx, &f) in bundle.feature_indices.iter().enumerate():
        bin = quantized.get_bin(f, row)
        if bin != missing_bin:
          bundled_bin = bundle.encode(local_idx, bin)
          break  // First non-missing wins (mutual exclusivity)
      
      data.push(bundled_bin)
  
  return BundledMatrix { data, bundles: feature_bundles }
```

## Integration with Histogram Building

### Modified Accumulation

Histogram building operates on bundles instead of original features:

```text
accumulate_bundled(histogram, bundled, grads, hess, rows):
  for bundle_idx in 0..bundled.n_bundles():
    hist = histogram.bundle_mut(bundle_idx)  // Larger histogram (sum of bundled bins)
    
    for row in rows:
      bin = bundled.get_bin(bundle_idx, row)
      hist.add(bin, grads[row], hess[row])
```

### Split Finding with Bundles

Split finding must decode bundle bins back to original features:

```text
find_best_split_bundled(histogram, bundled, bundle_idx, params):
  bundle = bundled.bundle(bundle_idx)
  best = SplitInfo::invalid()
  
  // Find best split within each original feature
  for (local_idx, &orig_feature) in bundle.feature_indices.iter().enumerate():
    offset = bundle.bin_offsets[local_idx]
    n_bins = bundle.n_bins_for_feature(local_idx)
    
    // Extract feature's histogram slice
    feature_hist = histogram.slice(offset, offset + n_bins)
    
    // Standard split enumeration on this slice
    split = find_best_numerical_split(feature_hist, params)
    split.feature = orig_feature  // Map back to original index
    split.bin = split.bin - offset  // Map back to original bin
    
    if split.gain > best.gain:
      best = split
  
  return best
```

## When to Use EFB

EFB is beneficial when:

```text
should_use_efb(quantized, cuts):
  // Compute overall sparsity
  total_nnz = 0
  total_values = quantized.n_rows() * quantized.n_features()
  
  for f in 0..quantized.n_features():
    for row in 0..quantized.n_rows():
      if quantized.get_bin(f, row) != missing_bin:
        total_nnz += 1
  
  sparsity = 1.0 - (total_nnz as f32 / total_values as f32)
  
  // EFB helps when data is sparse
  if sparsity < 0.5:
    return false  // Dense data: EFB overhead > benefit
  
  // Check if bundling reduces feature count significantly
  bundles = compute_bundles(quantized, cuts, max_conflict_rate=0.0)
  reduction = 1.0 - (bundles.len() as f32 / quantized.n_features() as f32)
  
  return reduction > 0.3  // At least 30% reduction
```

## Design Decisions

### DD-1: Greedy vs Optimal Bundling

**Context**: Finding optimal bundles is NP-hard (graph coloring).

**Decision**: Use greedy algorithm with feature ordering heuristic.

**Rationale**:

- Greedy achieves near-optimal results for typical sparse data
- O(n² features) one-time cost is acceptable
- Optimal would require exponential time
- LightGBM uses same approach

### DD-2: Conflict Rate Threshold

**Context**: Strict mutual exclusivity vs allowing some conflicts.

**Decision**: Configurable `max_conflict_rate` (default: 0.0).

**Rationale**:

- 0.0 = no conflicts = no accuracy loss
- Small rate (0.01-0.05) = more bundling, negligible accuracy impact
- User can tune based on accuracy/speed tradeoff

### DD-3: Bundle at Quantization Time

**Context**: When to compute bundles?

**Decision**: After quantization, before training.

**Rationale**:

- Quantized data reveals true sparsity (missing bins)
- One-time cost amortized over all boosting rounds
- Bundles remain fixed throughout training

### DD-4: First Non-Missing Wins

**Context**: How to handle conflicts (both features non-missing)?

**Decision**: First feature in bundle takes precedence.

**Rationale**:

- Simple, deterministic encoding
- With max_conflict_rate=0, conflicts are impossible
- With small conflict rate, order doesn't significantly affect accuracy
- Matches LightGBM behavior

## Performance Impact

| Scenario | Features | Sparsity | Bundles | Speedup |
|----------|----------|----------|---------|---------|
| One-hot (100 cats) | 100 | 99% | ~10 | ~10× |
| Text features | 10000 | 99.9% | ~200 | ~50× |
| Mixed dense+sparse | 500 | 50% | ~300 | ~1.7× |
| Dense data | 100 | 10% | ~100 | 1× (no benefit) |

## Future Work

- Incremental bundle updates for streaming data
- GPU-optimized bundled histogram kernel
- Automatic conflict rate tuning

## References

- [LightGBM Paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree) - Section 4.2
- [LightGBM EFB Implementation](https://github.com/microsoft/LightGBM/blob/master/src/io/dataset.cpp)
```
