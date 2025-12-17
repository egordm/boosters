# Categorical Feature Handling

## ELI5

Imagine you're sorting fruits. With numbers (like weight), you can say "heavier than 100g goes left, lighter goes right." But with categories (like apple, banana, orange), there's no natural order—you can't say "bigger than banana."

**Categorical handling** solves this by asking: "Which fruits should go left?" Maybe {apple, orange} go left and {banana} goes right. The tree learns the best grouping.

## ELI-Grad

Categorical features (discrete values without natural ordering) require special split mechanisms because threshold-based splits are meaningless:

**Numerical feature (ordered)**:
$$\text{Split: } x_j \le t \rightarrow \text{left, otherwise right}$$

**Categorical feature (unordered)**:
$$\text{Split: } x_j \in S \rightarrow \text{left, otherwise right}$$

where $S \subset \{\text{all categories}\}$ is a subset of categories.

### The Combinatorial Challenge

For a categorical feature with $k$ categories, there are $2^{k-1} - 1$ possible binary partitions (non-empty, non-trivial subsets up to symmetry).

| Categories | Possible Partitions |
|------------|---------------------|
| 3 | 3 |
| 5 | 15 |
| 10 | 511 |
| 20 | 524,287 |
| 100 | ~$10^{29}$ |

Exhaustive search is only feasible for small $k$. Larger cardinalities require approximation algorithms.

## Approaches to Categorical Splits

### Approach 1: One-Hot Encoding (External Preprocessing)

Convert each category to a binary feature:

```
Original: color ∈ {red, blue, green}

One-hot encoded:
  color_red:   [1, 0, 0] for red
  color_blue:  [0, 1, 0] for blue  
  color_green: [0, 0, 1] for green
```

**Advantages**:
- Works with any tree implementation (no special support needed)
- Each split tests one category ("is color = red?")
- Simple and well-understood

**Disadvantages**:
- Feature explosion: k categories → k features
- Memory inefficient for high cardinality
- Tree depth increases (multiple splits needed to test subsets)
- Loses relationships between categories

### Approach 2: Native Categorical Splits

Trees directly support partition-based splits:

```
Split: color ∈ {red, green} → left, {blue} → right
```

**Advantages**:
- No feature expansion
- More expressive splits (arbitrary subsets in one split)
- Compact model representation
- Can capture "similar category" relationships

**Disadvantages**:
- Exponential split candidates ($2^{k-1}$)
- Requires specialized algorithms
- Additional storage for category sets

## Split Finding Algorithms

### Algorithm 1: Exhaustive Search (Small Cardinality)

For few categories (k ≤ 4-8), enumerate all possible binary partitions:

```
ALGORITHM: ExhaustiveCategoricalSplit(categories, gradients, hessians)
----------------------------------------------------------------------
1. k <- |categories|
2. best_gain <- -infinity
3. best_partition <- null
4. 
5. // Enumerate all non-empty subsets (up to complement symmetry)
6. FOR subset IN all_subsets(categories) WHERE 0 < |subset| < k:
7.     // Compute gradient/hessian sums for left (subset) and right (complement)
8.     G_left, H_left <- sum_stats(subset, gradients, hessians)
9.     G_right, H_right <- sum_stats(complement, gradients, hessians)
10.    
11.    gain <- compute_split_gain(G_left, H_left, G_right, H_right)
12.    
13.    IF gain > best_gain:
14.        best_gain <- gain
15.        best_partition <- subset
16.
17. RETURN best_partition, best_gain
```

**Complexity**: $O(2^k)$ — exponential, only feasible for small k.

### Algorithm 2: One-Hot Strategy (LightGBM)

For k ≤ `max_cat_to_onehot` (default 4), test each category individually:

```
ALGORITHM: OneHotCategoricalSplit(categories, gradients, hessians)
------------------------------------------------------------------
1. best_gain <- -infinity
2. best_category <- null
3. 
4. FOR c IN categories:
5.     // Compute gain for splitting {c} vs {everything else}
6.     G_c, H_c <- sum_stats({c}, gradients, hessians)
7.     G_rest, H_rest <- sum_stats(categories - {c}, gradients, hessians)
8.     
9.     gain <- compute_split_gain(G_c, H_c, G_rest, H_rest)
10.    
11.    IF gain > best_gain:
12.        best_gain <- gain
13.        best_category <- c
14.
15. RETURN {best_category}, best_gain
```

This finds the single best category to isolate but won't find multi-category partitions like {red, green} vs {blue}.

**Complexity**: O(k) per split—linear in number of categories.

### Algorithm 3: Gradient-Sorted Strategy (LightGBM)

For k > `max_cat_to_onehot`, LightGBM uses a clever approximation based on a classic result:

**Key Insight (Fisher, 1958)**: For squared error loss, the optimal binary partition can be found by:
1. Sort categories by their mean target value
2. Test all O(k) split points in this sorted order

For gradient boosting, we use the gradient/hessian ratio as a proxy for "mean target":

```
ALGORITHM: GradientSortedCategoricalSplit(categories, gradients, hessians)
--------------------------------------------------------------------------
1. // Step 1: Compute per-category gradient ratio
2. FOR c IN categories:
3.     G_c <- sum(gradients[i] for i where x[i] = c)
4.     H_c <- sum(hessians[i] for i where x[i] = c)
5.     ratio[c] <- G_c / (H_c + smoothing)  // smoothing prevents division by zero
6. 
7. // Step 2: Sort categories by ratio
8. sorted_cats <- SORT(categories, key=ratio)
9. 
10. // Step 3: Find optimal split point in sorted order
11. best_gain <- -infinity
12. G_left, H_left <- 0, 0
13. G_total, H_total <- sum(gradients), sum(hessians)
14.
15. FOR i FROM 1 TO k-1:
16.    c <- sorted_cats[i-1]
17.    G_left += G_c[c]
18.    H_left += H_c[c]
19.    G_right <- G_total - G_left
20.    H_right <- H_total - H_left
21.    
22.    gain <- compute_split_gain(G_left, H_left, G_right, H_right)
23.    
24.    IF gain > best_gain:
25.        best_gain <- gain
26.        best_split_point <- i
27.
28. RETURN sorted_cats[0:best_split_point], best_gain
```

**Why This Works**: Categories with similar gradient/hessian ratios have similar optimal leaf values and should be grouped together. Sorting by ratio orders categories by their "effect direction," and the optimal partition is contiguous in this ordering.

**Complexity**: O(k log k) for sorting + O(k) for scanning = O(k log k)

> **Reference**: `LightGBM/src/treelearner/feature_histogram.cpp`, Fisher, W.D. (1958), "On Grouping for Maximum Homogeneity"

### Algorithm Comparison

| Algorithm | Complexity | Optimality | Use When |
|-----------|------------|------------|----------|
| Exhaustive | O(2^k) | Optimal | k ≤ 4 |
| One-hot | O(k) | Suboptimal | Finding single best category |
| Gradient-sorted | O(k log k) | Near-optimal for squared loss | k > 4 |

## Storage: Bitsets for Category Sets

### Representing Partitions

Store the "goes left" category set as a bitset:

```
Categories: {0: apple, 1: banana, 2: cherry, 3: date}
Split: {apple, cherry} go left

Bitset representation: 0b0101 = 5
        ||||
        |||+-- apple (0): 1 = goes left
        ||+--- banana (1): 0 = goes right  
        |+---- cherry (2): 1 = goes left
        +----- date (3): 0 = goes right

Decision: IF bitset & (1 << category) THEN left ELSE right
```

### Compact Bitset Storage

Pack bits into machine words:

```
For up to 64 categories: 1 uint64 word
For up to 128 categories: 2 uint64 words
For k categories: ceil(k / 64) uint64 words

Storage per split node: ceil(max_category / 64) * 8 bytes
```

### Variable-Size Storage

Different splits may have different cardinalities. Use CSR-like storage:

```
Categorical split storage:
  data:    [word0, word1, word2, word3, ...]  // All bitsets packed
  offsets: [0, 1, 3, 3, 5, ...]               // Start index per node
  sizes:   [1, 2, 0, 2, ...]                  // Words per node (0 = numerical split)

Access bitset for node i:
  start <- offsets[i]
  len <- sizes[i]
  bitset <- data[start : start + len]
```

### Memory Requirements

| Max Categories | Words per Split | Bytes per Split |
|----------------|-----------------|-----------------|
| 64 | 1 | 8 |
| 256 | 4 | 32 |
| 1,024 | 16 | 128 |
| 10,000 | 157 | 1,256 |

For high-cardinality features, consider:
- Limiting to top-k most frequent categories
- Hashing to fixed range
- Rejecting categorical split if cardinality exceeds threshold

## Decision Logic

### Traversal with Categorical Splits

```
ALGORITHM: TraverseWithCategorical(node, features, tree)
--------------------------------------------------------
1. WHILE node is not leaf:
2.     feat_idx <- tree.split_feature[node]
3.     fval <- features[feat_idx]
4.     
5.     IF tree.is_categorical[node]:
6.         // Categorical decision: check bitset membership
7.         category <- INT(fval)
8.         bitset <- tree.get_category_bitset(node)
9.         
10.        IF bitset.contains(category):
11.            node <- tree.left_child[node]
12.        ELSE:
13.            node <- tree.right_child[node]
14.    ELSE:
15.        // Numerical decision: threshold comparison
16.        IF fval <= tree.threshold[node]:
17.            node <- tree.left_child[node]
18.        ELSE:
19.            node <- tree.right_child[node]
20.
21. RETURN tree.leaf_value[node]
```

### Decision Type Encoding (LightGBM)

LightGBM packs split metadata into a single byte:

```
decision_type byte layout:
  Bit 0: Categorical flag     (0=numerical, 1=categorical)
  Bit 1: Default left flag    (0=missing→right, 1=missing→left)
  Bit 2-3: Missing type       (00=None, 01=Zero, 10=NaN)

Example: decision_type = 0b00000101 = 5
  - Bit 0: 1 → categorical split
  - Bit 1: 0 → missing values go right
  - Bit 2-3: 01 → treat 0.0 as missing
```

> **Reference**: `LightGBM/include/LightGBM/tree.h`, `kCategoricalMask`, `kDefaultLeftMask`

## Regularization for Categorical Splits

Categorical splits are prone to overfitting, especially with high cardinality (can perfectly separate small groups). Apply extra regularization:

### Additional L2 Penalty

```
Standard leaf value: w = -G / (H + lambda)

With categorical penalty: w = -G / (H + lambda + cat_l2)

LightGBM default: cat_l2 = 10.0
```

### Smoothing in Ratio Computation

Prevent extreme ratios for rare categories:

```
Standard ratio: ratio = G / H
Smoothed ratio: ratio = G / (H + cat_smooth)

LightGBM default: cat_smooth = 10.0
```

### Minimum Samples per Category

Skip categories with insufficient data:

```
IF count[category] < min_data_per_group:
    exclude category from split finding

LightGBM default: min_data_per_group = 100
```

### Maximum Categories per Split

Limit complexity of category sets:

```
Only consider top max_cat_threshold categories by |G/H| magnitude

LightGBM default: max_cat_threshold = 32
```

## When to Use Each Approach

### Use One-Hot Encoding When:
- Cardinality is low (< 10 categories)
- Using a library without native categorical support
- Interpretability is important (one feature per category)
- Categories are expected to have independent effects

### Use Native Categorical When:
- Cardinality is moderate (10-1000 categories)
- Memory efficiency matters
- Categories should be grouped (e.g., similar products, related locations)
- Using LightGBM, XGBoost (with `enable_categorical`), or compatible implementation

### Consider Alternatives When:
- Very high cardinality (> 1000): Target encoding, embeddings
- Ordinal relationship exists: Treat as numerical
- Too few samples per category: Group rare categories into "other"

## Encoding Consistency

**Critical**: Training and inference must use identical category → integer mappings.

### The Problem

```
Training data categories: ["apple", "banana", "cherry"]
Mapping: apple→0, banana→1, cherry→2

New inference data: ["cherry", "date", "apple"]
If different mapping: cherry→0, date→1, apple→2  // WRONG!
```

The model learned that "0" means apple, but inference is sending cherry as 0.

### Solution: Store Encoder with Model

```
Model artifact should include:
  1. Tree ensemble (splits, leaves)
  2. Category encoder (category string → integer mapping per feature)
  3. Feature metadata (which features are categorical)

At inference:
  1. Load encoder
  2. Transform input categories using stored mapping
  3. Handle unknown categories (default direction or error)
```

### Handling Unknown Categories

Options for categories seen at inference but not during training:

| Strategy | Behavior | Use When |
|----------|----------|----------|
| **Default direction** | Go right (or configured default) | Production systems, graceful degradation |
| **Error** | Reject input | Safety-critical, want to catch data issues |
| **Map to "other"** | Reserve special category during training | Expected unknown categories |

## Library Comparison

| Feature | LightGBM | XGBoost |
|---------|----------|---------|
| Native support | Yes | Yes (newer versions, `enable_categorical`) |
| Split algorithm | Gradient-sorted | Partition-based |
| One-hot threshold | Configurable (`max_cat_to_onehot=4`) | Fixed strategy |
| Max cardinality | Configurable | Limited |
| Bitset storage | Separate array | In-node |
| Linear tree compat | Yes | No |

### LightGBM Configuration

```python
params = {
    'categorical_feature': [0, 3, 7],  # Column indices of categorical features
    'max_cat_to_onehot': 4,            # Below this, use one-hot-like search
    'cat_l2': 10.0,                    # Extra L2 regularization
    'cat_smooth': 10.0,                # Ratio smoothing
    'max_cat_threshold': 32,           # Max categories per split
    'min_data_per_group': 100,         # Min samples per category
}
```

### XGBoost Configuration

```python
# Enable categorical support
dtrain = xgb.DMatrix(data, enable_categorical=True)

params = {
    'tree_method': 'hist',   # Required for categorical
    'max_cat_to_onehot': 4,  # One-hot threshold
}
```

## Summary

| Aspect | One-Hot Encoding | Native Categorical |
|--------|------------------|-------------------|
| **Preprocessing** | Required (k → k features) | Not needed |
| **Memory** | High (sparse binary features) | Low (single feature + bitsets) |
| **Split expressiveness** | Single category per split | Arbitrary subsets |
| **Algorithm** | Standard threshold | Gradient-sorted or exhaustive |
| **Tree depth** | Deeper (multiple splits for subset) | Shallower |
| **Overfitting risk** | Lower | Higher (needs regularization) |
| **Implementation** | Universal | Requires library support |

### Key References

- Fisher, W.D. (1958), "On Grouping for Maximum Homogeneity" — Mathematical foundation
- `LightGBM/src/treelearner/feature_histogram.cpp` — Gradient-sorted split finding
- `LightGBM/src/io/bin.cpp` — Category binning
- `LightGBM/include/LightGBM/tree.h` — Categorical split storage and decision
- `xgboost/include/xgboost/tree_model.h` — Categorical split in XGBoost
