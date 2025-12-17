# LightGBM Data Structures

> Source: `include/LightGBM/tree.h`, `include/LightGBM/bin.h`, `include/LightGBM/dataset.h`

## Overview

This document covers LightGBM's core data structures for inference and their implications for booste-rs.

## Tree Structure

### Memory Layout (Structure of Arrays)

LightGBM uses **SoA (Structure of Arrays)** layout, storing each field in a separate vector:

```cpp
class Tree {
  // Split node data (size = num_leaves - 1)
  std::vector<int> left_child_;
  std::vector<int> right_child_;
  std::vector<int> split_feature_inner_;   // Internal feature index
  std::vector<int> split_feature_;          // Original feature index
  std::vector<uint32_t> threshold_in_bin_;  // For binned data
  std::vector<double> threshold_;           // For raw data
  std::vector<int8_t> decision_type_;       // Packed flags
  std::vector<float> split_gain_;
  
  // Leaf data (size = num_leaves)
  std::vector<int> leaf_parent_;
  std::vector<double> leaf_value_;
  std::vector<double> leaf_weight_;
  std::vector<int> leaf_count_;
  std::vector<int> leaf_depth_;
  
  // Internal node statistics (size = num_leaves - 1)
  std::vector<double> internal_value_;
  std::vector<double> internal_weight_;
  std::vector<int> internal_count_;
  
  // Categorical data (variable size)
  int num_cat_;
  std::vector<int> cat_boundaries_;
  std::vector<uint32_t> cat_threshold_;      // Bitset for categories
  std::vector<int> cat_boundaries_inner_;
  std::vector<uint32_t> cat_threshold_inner_;
  
  // Linear tree data (optional, size = num_leaves)
  bool is_linear_;
  std::vector<std::vector<double>> leaf_coeff_;
  std::vector<double> leaf_const_;
  std::vector<std::vector<int>> leaf_features_;
  std::vector<std::vector<int>> leaf_features_inner_;
  
  // Metadata
  int num_leaves_;
  int max_leaves_;
  double shrinkage_;
  int max_depth_;
};
```

### Node Indexing Convention

LightGBM uses a **negative index** convention for leaves:

```text
Internal nodes: 0, 1, 2, ... (non-negative)
Leaf nodes:     ~0, ~1, ~2, ... (negative, using bitwise NOT)

Example tree with 3 leaves:
         [0]           <- Internal node 0
        /   \
      [~0]  [1]        <- Leaf 0, Internal node 1
           /   \
         [~1]  [~2]    <- Leaf 1, Leaf 2

left_child_ = [~0, ~1]   (node 0 → leaf 0, node 1 → leaf 1)
right_child_ = [1, ~2]   (node 0 → node 1, node 1 → leaf 2)
```

### Decision Type Encoding

All split metadata packed into single byte:

```text
Bit 0: Categorical flag (0 = numerical, 1 = categorical)
Bit 1: Default left flag (0 = missing goes right, 1 = missing goes left)
Bits 2-3: Missing type
  00 (0): None - no special missing handling
  01 (1): Zero - treat 0.0 as missing
  10 (2): NaN  - treat NaN as missing

Example: decision_type_ = 0b00001010 = 10
  - Categorical: 0 (numerical)
  - Default left: 1 (yes)
  - Missing type: 2 (NaN)
```

### Categorical Split Storage

Categories stored as **bitsets** for efficient membership testing:

```cpp
// For a categorical split at node i:
int cat_idx = threshold_[i];  // Index into cat_boundaries_
uint32_t* bitset_start = cat_threshold_.data() + cat_boundaries_[cat_idx];
int bitset_len = cat_boundaries_[cat_idx + 1] - cat_boundaries_[cat_idx];

// Test if category c is in the "go left" set:
bool go_left = Common::FindInBitset(bitset_start, bitset_len, c);
```

## Comparison with XGBoost

### XGBoost Tree Node (Array of Structures)

```cpp
// XGBoost: 24 bytes per node (AoS)
class Node {
  int32_t parent_;   // Parent + is_left_child in high bit
  int32_t cleft_;    // Left child (kInvalidNodeId if leaf)
  int32_t cright_;   // Right child
  uint32_t sindex_;  // Split index + default_left in high bit
  union Info {
    float leaf_value;
    float split_cond;
  } info_;
};

// Categorical stored separately
std::vector<uint32_t> split_categories_;
std::vector<RegTree::Segment> split_categories_segments_;
```

### Layout Comparison

| Aspect | LightGBM (SoA) | XGBoost (AoS) |
|--------|----------------|---------------|
| **Node size** | Variable (many arrays) | Fixed 24 bytes |
| **Cache locality** | Good for single field | Good for full node |
| **SIMD friendly** | Yes (contiguous arrays) | No |
| **Memory overhead** | Higher (vector overhead) | Lower |
| **Leaf encoding** | Negative indices | IsLeaf() check |
| **Threshold type** | double (64-bit) | float (32-bit) |
| **Feature index** | Both inner and raw | Single index |

### Memory Estimates

For a tree with N leaves (N-1 internal nodes):

**LightGBM:**
```
Split nodes: (N-1) × (4+4+4+4+4+8+1+4) = (N-1) × 33 bytes
Leaf nodes: N × (4+8+8+4+4) = N × 28 bytes
Categorical: Variable
Linear: Variable
Total base: ~61N bytes
```

**XGBoost:**
```
All nodes: (2N-1) × 24 = ~48N bytes
Stats: (2N-1) × 16 = ~32N bytes
Total: ~80N bytes
```

## BinMapper (Feature Quantization)

Maps continuous values to discrete bins:

```cpp
class BinMapper {
  int num_bin_;
  MissingType missing_type_;
  std::vector<double> bin_upper_bound_;    // Numerical bins
  std::unordered_map<int, uint32_t> categorical_2_bin_;  // Category → bin
  std::vector<int> bin_2_categorical_;     // Bin → category
  BinType bin_type_;
  double min_val_, max_val_;
  uint32_t default_bin_;      // Bin for value 0
  uint32_t most_freq_bin_;    // Most frequent bin
  double sparse_rate_;
  bool is_trivial_;
};
```

### Value to Bin Conversion

```cpp
inline uint32_t BinMapper::ValueToBin(double value) const {
  if (bin_type_ == BinType::NumericalBin) {
    // Binary search in bin_upper_bound_
    if (std::isnan(value)) {
      if (missing_type_ == MissingType::NaN) {
        return num_bin_ - 1;  // NaN bin is last
      }
      value = 0.0f;
    }
    // ... binary search
  } else {
    // Categorical: direct lookup
    int int_val = static_cast<int>(value);
    auto it = categorical_2_bin_.find(int_val);
    if (it != categorical_2_bin_.end()) {
      return it->second;
    }
    return 0;  // Unknown category
  }
}
```

## Dataset Storage

### Bin Storage Types

```cpp
// Dense storage: simple array
template <typename VAL_T>  // uint8_t, uint16_t, uint32_t
class DenseBin {
  std::vector<VAL_T> data_;  // data_[row] = bin_idx
  
  uint32_t Get(data_size_t idx) {
    return data_[idx];
  }
};

// Sparse storage: delta-encoded
template <typename VAL_T>
class SparseBin {
  std::vector<VAL_T> vals_;           // Non-zero bin values
  std::vector<data_size_t> deltas_;   // Row index deltas
  
  // Requires iteration to access specific row
};
```

### Feature Groups

LightGBM groups features for cache efficiency:

```cpp
class FeatureGroup {
  std::vector<std::unique_ptr<Bin>> bin_data_;
  std::vector<std::unique_ptr<BinMapper>> bin_mappers_;
  bool is_multi_val_;  // Multi-value bin (EFB)
  // ...
};
```

## Linear Tree Data

Additional per-leaf storage for linear models:

```cpp
// Per leaf (size = num_leaves):
std::vector<double> leaf_const_;           // Intercept term
std::vector<std::vector<double>> leaf_coeff_;  // Coefficients
std::vector<std::vector<int>> leaf_features_;  // Feature indices (raw)
std::vector<std::vector<int>> leaf_features_inner_;  // Feature indices (inner)
```

### Memory for Linear Trees

Per leaf with k features:
- `leaf_const_`: 8 bytes
- `leaf_coeff_`: 8k bytes + vector overhead
- `leaf_features_`: 4k bytes + vector overhead
- `leaf_features_inner_`: 4k bytes + vector overhead

Total: ~20k + 48 bytes per leaf (with vector overhead)

## Serialization Format

### Text Format

```
num_leaves=5
num_cat=1
split_feature=0 2 1 3
split_gain=0.5 0.3 0.2 0.1
threshold=1.5 2.5 0.5 3.5
decision_type=0 2 1 0
left_child=-1 -2 -3 -4
right_child=1 2 3 -5
leaf_value=0.1 0.2 0.3 0.4 0.5
...
is_linear=1
leaf_const=0.1 0.2 0.3 0.4 0.5
num_features=2 1 0 2 1
leaf_features=0 1  2  0 3  1
leaf_coeff=0.5 0.3  0.2  0.1 0.4  0.2
```

### JSON Format

```json
{
  "num_leaves": 3,
  "num_cat": 0,
  "shrinkage": 0.1,
  "tree_structure": {
    "split_index": 0,
    "split_feature": 2,
    "split_gain": 0.5,
    "threshold": 1.5,
    "decision_type": "<=",
    "default_left": true,
    "missing_type": "NaN",
    "left_child": {
      "leaf_index": 0,
      "leaf_value": 0.1,
      "leaf_const": 0.05,
      "leaf_features": [2, 3],
      "leaf_coeff": [0.1, 0.2]
    },
    "right_child": { ... }
  }
}
```

## booste-rs Recommendations

### Tree Structure

```rust
/// LightGBM-style tree with SoA layout
pub struct LgbTree {
    // Split nodes
    pub left_child: Box<[i32]>,
    pub right_child: Box<[i32]>,
    pub split_feature: Box<[u32]>,
    pub threshold: Box<[f64]>,
    pub decision_type: Box<[u8]>,
    
    // Leaf values
    pub leaf_value: Box<[f64]>,
    
    // Categorical (optional)
    pub cat_boundaries: Option<Box<[u32]>>,
    pub cat_threshold: Option<Box<[u32]>>,
    
    // Linear tree (optional)
    pub linear: Option<LinearTreeData>,
    
    // Metadata
    pub num_leaves: u32,
    pub shrinkage: f64,
}

pub struct LinearTreeData {
    pub leaf_const: Box<[f64]>,
    pub leaf_coeff: Vec<Box<[f64]>>,     // Per-leaf coefficients
    pub leaf_features: Vec<Box<[u32]>>,  // Per-leaf feature indices
}
```

### Decision Type Helpers

```rust
impl LgbTree {
    #[inline]
    fn is_categorical(&self, node: usize) -> bool {
        (self.decision_type[node] & 0x1) != 0
    }
    
    #[inline]
    fn default_left(&self, node: usize) -> bool {
        (self.decision_type[node] & 0x2) != 0
    }
    
    #[inline]
    fn missing_type(&self, node: usize) -> MissingType {
        match (self.decision_type[node] >> 2) & 0x3 {
            0 => MissingType::None,
            1 => MissingType::Zero,
            2 => MissingType::NaN,
            _ => MissingType::None,
        }
    }
}
```

### Unified Tree Interface

Consider a unified interface that works with both XGBoost and LightGBM formats:

```rust
pub trait TreePredictor {
    fn predict(&self, features: &[f32]) -> f32;
    fn predict_leaf(&self, features: &[f32]) -> i32;
    fn num_leaves(&self) -> usize;
    fn is_linear(&self) -> bool;
}

impl TreePredictor for LgbTree { ... }
impl TreePredictor for XgbTree { ... }
```

## Source References

| Component | File |
|-----------|------|
| Tree header | `include/LightGBM/tree.h` |
| Tree impl | `src/io/tree.cpp` |
| BinMapper | `include/LightGBM/bin.h`, `src/io/bin.cpp` |
| Dataset | `include/LightGBM/dataset.h` |
| FeatureGroup | `include/LightGBM/feature_group.h` |
