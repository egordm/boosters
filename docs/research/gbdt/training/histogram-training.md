# Histogram-Based Training

Histogram-based training is the core algorithmic innovation that makes modern gradient
boosting fast and scalable. Instead of considering every unique feature value as a
potential split point, we discretize features into bins and aggregate gradient statistics
per bin.

---

## The Core Insight

Traditional split finding examines every unique value:

```text
Sort feature values: [1.2, 2.3, 3.7, 5.1, 8.9, ...]  ← O(n log n)
Try each as split:   for each value, compute gain    ← O(n)
```

Histogram-based approach:

```text
Bin boundaries: [0, 2, 4, 6, 8, 10]
Count gradients per bin: bin[0]: sum_g=5.2, sum_h=3.1
                         bin[1]: sum_g=2.1, sum_h=1.8
                         ...
Try each bin as split: for each bin, compute gain   ← O(bins) where bins << n
```

For a candidate split at bin boundary $b$, we partition samples into left ($I_L$) and
right ($I_R$) sets. The split gain is:

$$
\text{Gain} = \frac{1}{2}\left[
  \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}
\right] - \gamma
$$

Where $G_L = \sum_{i \in I_L} g_i$ and $H_L = \sum_{i \in I_L} h_i$ are the gradient
and Hessian sums for the left partition.

**Key insight**: We only need $G$ and $H$ sums, not individual gradients. This enables
aggregation into histograms.

---

## Why Histograms?

| Approach | Split Finding | Memory | Distributed |
|----------|---------------|--------|-------------|
| Exact (sort-based) | O(n log n) | O(n) floats | Hard |
| Histogram-based | O(bins) | O(bins) | Easy (merge histograms) |

Typical values: n = 1,000,000 samples, bins = 256.

Speedup: ~4000× for split finding, 4× memory reduction.

---

## Building Histograms

### Algorithm

```
Algorithm: Build Histogram for a Node
─────────────────────────────────────────
Input: 
  - Quantized features X_bin (n × d matrix of bin indices)
  - Gradients g, Hessians h (n-vectors)
  - Sample indices I belonging to this node

Output: Histogram H[f][b] = (sum_g, sum_h) for each feature f and bin b

for each sample i in I:
    for each feature f:
        bin = X_bin[i, f]
        H[f][bin].sum_g += g[i]
        H[f][bin].sum_h += h[i]
```

### Row-wise vs Column-wise

**Row-wise** (iterate over samples, then features):

- Better when #samples << #features
- More cache-friendly for row-major data
- Used by: XGBoost (default), LightGBM (adaptive)

**Column-wise** (iterate over features, then samples):

- Better when #features << #samples
- Enables feature-parallel building
- Used by: LightGBM (adaptive)

---

## Histogram Subtraction Trick

For a binary split, we have: `parent = left_child + right_child`

Therefore: `sibling = parent - built_child`

**Algorithm**:

1. Build histogram for the **smaller** child (fewer samples)
2. Compute sibling histogram by subtraction
3. Save ~50% of histogram building work

This optimization is critical for efficiency. Both XGBoost and LightGBM implement it.

```
Cost Analysis:
─────────────────────────────────────────
Without subtraction: O(n_left) + O(n_right) = O(n_parent)
With subtraction:    O(min(n_left, n_right)) + O(bins)

Savings: Nearly 50% on average
```

---

## Parallel Histogram Building

Histograms can be built in parallel using different strategies:

### Data-Parallel (XGBoost approach)

1. Partition samples across threads
2. Each thread builds local histogram
3. Reduce (sum) local histograms into global

**Trade-off**: Requires O(threads × bins × features) memory for local histograms.

### Feature-Parallel

1. Partition features across threads
2. Each thread builds histograms for its features
3. No reduction needed

**Trade-off**: Requires synchronized access to sample indices.

---

## Gradient Discretization (LightGBM)

LightGBM can quantize gradients themselves for even faster histogram building:

| Precision | Storage | Histogram Entry |
|-----------|---------|-----------------|
| float64 (default) | 8 bytes | 16 bytes (g + h) |
| int16 packed | 2 bytes | 4 bytes |
| int8 packed | 1 byte | 2 bytes |

Benefits:

- 4-8× less memory bandwidth
- Faster histogram accumulation
- Slight accuracy trade-off

The discretization adapts based on leaf size — smaller leaves use lower precision
since they're less likely to overflow.

---

## Histogram Structure

A histogram entry stores gradient and Hessian sums for each bin:

```text
Feature histogram for feature f:
┌─────────────────────────────────────────────────┐
│ Bin 0: (sum_g = 1.5, sum_h = 0.8)              │
│ Bin 1: (sum_g = 2.3, sum_h = 1.2)              │
│ Bin 2: (sum_g = 0.1, sum_h = 0.3)              │
│ ...                                             │
│ Bin 255: (sum_g = 0.7, sum_h = 0.4)            │
└─────────────────────────────────────────────────┘
```

Memory layout considerations:

- **Interleaved (g, h, g, h, ...)**: Better for split finding (access both together)
- **Separate (g[], h[])**: Better for SIMD (process all g, then all h)

XGBoost and LightGBM use interleaved layout.

---

## Finding Splits from Histograms

Once histograms are built, finding the best split is O(bins):

```
Algorithm: Find Best Split for Feature f
─────────────────────────────────────────
Input: Histogram H[b] for b = 0..num_bins-1

1. Compute total: G_total = Σ H[b].sum_g, H_total = Σ H[b].sum_h

2. Scan left-to-right:
   G_left = 0, H_left = 0
   for b = 0 to num_bins-1:
       G_left += H[b].sum_g
       H_left += H[b].sum_h
       G_right = G_total - G_left
       H_right = H_total - H_left
       
       gain = compute_gain(G_left, H_left, G_right, H_right)
       if gain > best_gain:
           best_gain = gain
           best_threshold = b
```

This cumulative scan approach avoids recomputing sums for each candidate.

---

## Missing Value Handling

Missing values need special handling in histograms:

**XGBoost approach**: Try both directions

```
For each split:
  1. Try all missing → left: compute gain
  2. Try all missing → right: compute gain
  3. Choose better direction
```

**Implementation**: Scan histogram in both directions (forward and backward), tracking
best gain from each. Store the learned `default_left` direction with the split.

---

## Complexity Summary

| Operation | Time | Notes |
|-----------|------|-------|
| Histogram building | O(n × d) | Per node, n samples, d features |
| With subtraction | O(n/2 × d) | Build smaller child only |
| Split finding | O(bins × d) | Per node |
| Total per tree | O(n × d × depth) | Dominated by histogram building |

The key win is that `bins` (typically 256) is much smaller than `n` (often millions).

---

## Source References

### XGBoost

- `src/tree/hist/histogram.h` — Histogram building
- `src/tree/updater_quantile_hist.cc` — Main training loop
- `src/common/hist_util.cc` — Histogram utilities

### LightGBM

- `src/treelearner/feature_histogram.hpp` — Histogram structure
- `src/io/dense_bin.hpp` — Dense histogram building
- `src/treelearner/serial_tree_learner.cpp` — Training loop
