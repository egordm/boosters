# Split Finding

Split finding determines how to partition samples at each tree node. Given a set of
samples with their gradient and Hessian values, we find the feature and threshold that
maximizes information gain.

---

## The Split Gain Formula

### ELI5

When splitting a group of students into two rooms, we want each room to be as "similar"
as possible internally. A good split puts all the A students in one room and all the
C students in another. A bad split mixes them together in both rooms.

### ELI-Grad

The gain from a split measures improvement in the objective:

$$
\text{Gain} = \frac{1}{2}\left[
  \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda}
\right] - \gamma
$$

Where:
- $G_L, H_L$ = sum of gradients/Hessians in left partition
- $G_R, H_R$ = sum of gradients/Hessians in right partition  
- $G, H$ = sum of gradients/Hessians in parent node
- $\lambda$ = L2 regularization on leaf weights
- $\gamma$ = complexity penalty per split

**Intuition**: The formula compares "what we can achieve with two leaves" vs "what we
have with one leaf." The difference is the gain from splitting.

---

## Split Enumeration

For histogram-based training, we enumerate splits by scanning the histogram:

### Forward Scan (Cumulative Sum)

```text
Algorithm: Enumerate Splits for Feature f
─────────────────────────────────────────
Given histogram H[0..B-1] with (sum_g, sum_h) per bin

G_total = Σ H[b].sum_g
H_total = Σ H[b].sum_h

G_left = 0, H_left = 0
best_gain = -∞

for b = 0 to B-2:    # B-1 possible split points
    G_left += H[b].sum_g
    H_left += H[b].sum_h
    G_right = G_total - G_left
    H_right = H_total - H_left
    
    # Skip if too few samples on either side
    if H_left < min_child_weight or H_right < min_child_weight:
        continue
    
    gain = compute_gain(G_left, H_left, G_right, H_right)
    
    if gain > best_gain:
        best_gain = gain
        best_split = b
```

Time complexity: O(bins) per feature.

---

## Missing Value Handling

Real data often has missing values. GBDT can learn the best default direction.

### Bidirectional Scanning

```text
Algorithm: Find Best Split with Missing Values
─────────────────────────────────────────
Given histogram H[0..B-1] and missing_sum = (G_miss, H_miss)

# Option 1: Missing values go LEFT
Scan left-to-right, adding missing to left partition
Record best_gain_left

# Option 2: Missing values go RIGHT  
Scan right-to-left, adding missing to right partition
Record best_gain_right

if best_gain_left > best_gain_right:
    default_left = true
    best_gain = best_gain_left
else:
    default_left = false
    best_gain = best_gain_right
```

The learned `default_left` is stored with the split and used during inference.

---

## Constraints

### Minimum Child Weight

Prevent splits that create leaves with too few samples:

```text
if H_left < min_child_weight or H_right < min_child_weight:
    skip this split
```

Where Hessian sum approximates the effective sample count (exactly for MSE loss).

### Minimum Split Gain

Only accept splits with sufficient improvement:

```text
if gain < min_split_gain:
    don't split (make this a leaf)
```

The $\gamma$ parameter in the gain formula serves this purpose.

### Monotonic Constraints

Force predictions to be monotonically increasing/decreasing with a feature:

```text
Monotonic increasing constraint on feature f:
  Only allow splits where left_prediction ≤ right_prediction
  
Monotonic decreasing constraint on feature f:
  Only allow splits where left_prediction ≥ right_prediction
```

**Implementation**: After computing optimal leaf weights, verify the constraint is
satisfied before accepting the split.

### Interaction Constraints

Limit which features can appear together in paths:

```text
If feature A is used in an ancestor:
  Only allow features in same interaction group
```

This reduces model complexity and can improve interpretability.

---

## Leaf Weight Calculation

Once a split is chosen, we compute optimal leaf weights:

$$
w^* = -\frac{\sum_{i \in \text{leaf}} g_i}{\sum_{i \in \text{leaf}} h_i + \lambda}
$$

This is the Newton-Raphson update step, treating the leaf as a single "parameter."

For multi-output, each output dimension has its own weight using the same formula.

---

## Categorical Splits

For categorical features, the split isn't "< threshold" but "∈ category set":

```text
Standard split: feature < threshold → go left
Categorical split: feature ∈ {cat_a, cat_c, cat_e} → go left
```

Finding the optimal partition of k categories is exponential (2^k subsets).
LightGBM uses gradient-based sorting to find a good partition in O(k log k).

See [Categorical Features](../../categorical-features.md) for details.

---

## Parallelization

Split finding parallelizes naturally:

**Feature-parallel**: Each thread evaluates different features

```text
parallel for f in features:
    best_split[f] = find_best_split(histogram[f])

global_best = argmax(best_split)
```

**Within-feature**: For very wide features, parallelize the bin scan
(rarely needed with 256 bins).

---

## Numerical Stability

The gain formula can have numerical issues:

**Issue**: Division by small Hessian sums

**Solution**: The $\lambda$ term ensures denominator is never too small

**Issue**: Subtracting large similar numbers (catastrophic cancellation)

**Solution**: Use numerically stable formulations, or higher precision for accumulation

---

## Complexity

| Operation | Complexity |
|-----------|------------|
| Single feature | O(bins) |
| All features | O(features × bins) |
| With missing values | O(features × bins) — 2 scans but still linear |
| With constraints | O(features × bins) — extra checks per candidate |

The histogram approach keeps this fast regardless of sample count.

---

## Source References

### XGBoost

- `src/tree/split_evaluator.cc` — Split gain calculation
- `src/tree/hist/evaluate_splits.cc` — Split enumeration
- `include/xgboost/tree_model.h` — SplitEntry structure

### LightGBM

- `src/treelearner/feature_histogram.cpp` — Split finding from histograms
- `src/treelearner/split_info.hpp` — Split information structure
