# LightGBM Model Format Research

**Date**: 2024-11-30  
**Status**: Research complete for text format

---

## Overview

LightGBM supports multiple model serialization formats:
1. **Text format** (`.txt`) — Human-readable, well-documented, recommended for parsing
2. **Binary format** — Faster but undocumented internal format
3. **JSON dump** — Alternative output via `DumpModel()`, includes pretty-printed tree structure

This document focuses on the **text format** since it's the primary interchange format.

---

## Text Format Structure

The text format is line-based with `key=value` pairs. Structure:

```
<model_type>          # e.g., "tree"
version=v4
num_class=<int>
num_tree_per_iteration=<int>
label_index=<int>
max_feature_idx=<int>
objective=<string>    # optional
average_output        # flag, presence means true
feature_names=<space-separated list>
monotone_constraints=<space-separated list>  # optional
feature_infos=<space-separated list>

tree_sizes=<space-separated list>   # sizes of each tree serialization (for parallel loading)

Tree=0
<tree 0 content>

Tree=1
<tree 1 content>

...

end of trees

feature_importances:
<feature>=<importance>
...

parameters:
<config parameters>
end of parameters

parser:
<parser config>
end of parser
```

### Header Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Model format version, e.g., `v4` |
| `num_class` | int | Number of output classes (1 for regression, ≥2 for multiclass) |
| `num_tree_per_iteration` | int | Trees per boosting round (1 for regression/binary, num_class for multiclass) |
| `label_index` | int | Index of label column in training data |
| `max_feature_idx` | int | Maximum feature index used (0-based, so num_features = max_feature_idx + 1) |
| `objective` | string | Objective function string (e.g., `regression`, `binary`, `multiclass`) |
| `average_output` | flag | If present, average predictions across iterations |
| `feature_names` | string[] | Space-separated feature names |
| `feature_infos` | string[] | Space-separated feature metadata (min:max for numerical, values for categorical) |
| `tree_sizes` | int[] | Byte sizes of each tree's serialization (for parallel loading) |

### Objective String Format

From `src/boosting/gbdt_model_text.cpp` line 330:
```cpp
ss << "objective=" << objective_function_->ToString() << '\n';
```

Common objective strings:
- `regression` — Regression with L2 loss
- `regression_l1` — Regression with L1 loss
- `binary` — Binary classification with logloss
- `multiclass num_class:3` — Multiclass with softmax, 3 classes
- `multiclassova num_class:3` — One-vs-all multiclass

---

## Tree Format

Each tree is serialized as key-value lines. From `src/io/tree.cpp` `ToString()`:

```
num_leaves=<int>
num_cat=<int>                          # number of categorical splits
split_feature=<int[]>                  # feature index for each split node
split_gain=<float[]>                   # gain for each split
threshold=<double[]>                   # split threshold for each node
decision_type=<int[]>                  # bitfield encoding split type
left_child=<int[]>                     # left child index (negative = leaf)
right_child=<int[]>                    # right child index (negative = leaf)
leaf_value=<double[]>                  # output value for each leaf
leaf_weight=<double[]>                 # sum of hessians at each leaf
leaf_count=<int[]>                     # sample count at each leaf
internal_value=<double[]>              # internal node value (for SHAP)
internal_weight=<double[]>             # internal node weight
internal_count=<int[]>                 # internal node sample count
cat_boundaries=<int[]>                 # boundaries for categorical threshold bitset (if num_cat > 0)
cat_threshold=<uint32[]>               # categorical threshold bitset (if num_cat > 0)
is_linear=<0|1>                        # whether tree has linear models at leaves
shrinkage=<double>                     # learning rate applied to this tree
```

### Array Sizes

| Array | Size | Notes |
|-------|------|-------|
| `split_feature`, `split_gain`, `threshold`, `decision_type` | `num_leaves - 1` | One per internal node |
| `left_child`, `right_child` | `num_leaves - 1` | One per internal node |
| `leaf_value`, `leaf_weight`, `leaf_count` | `num_leaves` | One per leaf |
| `internal_value`, `internal_weight`, `internal_count` | `num_leaves - 1` | One per internal node |

### Node/Leaf Indexing Convention

**Critical difference from XGBoost:**

- Internal nodes: indexed `0` to `num_leaves - 2`
- Leaves: referenced as **negative values** in `left_child`/`right_child`
- Leaf index: `~child` (bitwise NOT) when `child < 0`

Example:
```
left_child=-1   → leaf index 0   (~(-1) = 0)
right_child=-3  → leaf index 2   (~(-3) = 2)
left_child=5    → internal node 5
```

From `tree.h` line 321:
```cpp
inline int Tree::GetLeaf(const double* feature_values) const {
  int node = 0;
  while (node >= 0) {
    node = NumericalDecision(feature_values[split_feature_[node]], node);
  }
  return ~node;  // Convert negative to leaf index
}
```

---

## Decision Type Bitfield

The `decision_type` field is an `int8_t` (serialized as int) encoding multiple flags:

```cpp
// From tree.h
#define kCategoricalMask (1)    // bit 0: 1 = categorical, 0 = numerical
#define kDefaultLeftMask (2)    // bit 1: 1 = default left, 0 = default right
// bits 2-3: missing type encoding
```

### Bit Layout

```
Bits: 7 6 5 4 | 3 2 | 1 | 0
              |     |   |
              |     |   +-- Categorical flag (0=numerical, 1=categorical)
              |     +------ Default left flag (0=right, 1=left)
              +------------ Missing type (0=None, 1=Zero, 2=NaN)
```

### Missing Type Values

From `tree.h` `GetMissingType()`:
```cpp
inline static int8_t GetMissingType(int8_t decision_type) {
  return (decision_type >> 2) & 3;  // bits 2-3
}
```

| Value | Enum | Meaning |
|-------|------|---------|
| 0 | `None` | No special missing handling |
| 1 | `Zero` | Treat zeros as missing |
| 2 | `NaN` | Treat NaN as missing |

### Parsing Example

```rust
fn parse_decision_type(dt: i8) -> (bool, bool, MissingType) {
    let is_categorical = (dt & 1) != 0;
    let default_left = (dt & 2) != 0;
    let missing_type = match (dt >> 2) & 3 {
        0 => MissingType::None,
        1 => MissingType::Zero,
        2 => MissingType::NaN,
        _ => unreachable!(),
    };
    (is_categorical, default_left, missing_type)
}
```

---

## Numerical Split Logic

From `tree.h` `NumericalDecision()`:

```cpp
inline int NumericalDecision(double fval, int node) const {
  uint8_t missing_type = GetMissingType(decision_type_[node]);
  
  // Convert NaN to 0 if missing_type != NaN
  if (std::isnan(fval) && missing_type != MissingType::NaN) {
    fval = 0.0f;
  }
  
  // Handle missing values
  if ((missing_type == MissingType::Zero && IsZero(fval))
      || (missing_type == MissingType::NaN && std::isnan(fval))) {
    if (GetDecisionType(decision_type_[node], kDefaultLeftMask)) {
      return left_child_[node];
    } else {
      return right_child_[node];
    }
  }
  
  // Standard comparison: left if fval <= threshold
  if (fval <= threshold_[node]) {
    return left_child_[node];
  } else {
    return right_child_[node];
  }
}
```

### Key Differences from XGBoost

1. **Condition**: LightGBM uses `<=` (left if value ≤ threshold), XGBoost uses `<`
2. **NaN handling**: LightGBM converts NaN to 0 by default unless `missing_type == NaN`
3. **Zero handling**: Can treat zeros as missing via `missing_type == Zero`

---

## Categorical Split Logic

From `tree.h` `CategoricalDecision()`:

```cpp
inline int CategoricalDecision(double fval, int node) const {
  int int_fval;
  if (std::isnan(fval)) {
    return right_child_[node];  // NaN always goes right
  } else {
    int_fval = static_cast<int>(fval);
    if (int_fval < 0) {
      return right_child_[node];  // Negative values go right
    }
  }
  int cat_idx = static_cast<int>(threshold_[node]);
  if (Common::FindInBitset(cat_threshold_.data() + cat_boundaries_[cat_idx],
                           cat_boundaries_[cat_idx + 1] - cat_boundaries_[cat_idx], int_fval)) {
    return left_child_[node];
  }
  return right_child_[node];
}
```

### Categorical Threshold Encoding

- `threshold[node]` stores an index into `cat_boundaries`
- `cat_boundaries[cat_idx]` to `cat_boundaries[cat_idx + 1]` defines a range in `cat_threshold`
- `cat_threshold` is a bitset of uint32 values
- Category `c` is in the set if bit `c % 32` of `cat_threshold[cat_boundaries[cat_idx] + c / 32]` is set

### Bitset Example

```
cat_boundaries = [0, 2]  # One categorical split using indices 0-1 in cat_threshold
cat_threshold = [5, 0]   # Bitset: 5 = 0b101 = categories {0, 2}
```

Categories in left child: `{0, 2}`

---

## Linear Trees

When `is_linear=1`, each leaf has a linear model instead of a constant:

```
is_linear=1
leaf_const=<double[]>           # bias term for each leaf
num_features=<int[]>            # number of features in each leaf's model
leaf_features=<int[] int[] ...> # feature indices for each leaf (space-separated groups)
leaf_coeff=<double[] double[] ...> # coefficients for each leaf (space-separated groups)
```

Prediction: `leaf_const[leaf] + sum(leaf_coeff[leaf][i] * features[leaf_features[leaf][i]])`

**Note**: Linear trees are out of scope for initial implementation.

---

## Multi-class Models

For multiclass with `K` classes:
- `num_class = K`
- `num_tree_per_iteration = K`
- Trees are organized as: `tree[iteration * K + class_idx]`

Raw prediction output is a vector of length K, then softmax is applied:
```python
raw_scores = [sum(trees[i * K + c].predict(x) for i in range(num_iterations)) for c in range(K)]
probs = softmax(raw_scores)
```

---

## Shrinkage (Learning Rate)

Each tree stores a `shrinkage` value (default 1.0). 

**Critical implementation note**: The `leaf_value` in the serialized model has **already been multiplied by the learning rate**. This is done during training by `Tree::Shrinkage()` in `tree.h`:

```cpp
virtual inline void Shrinkage(double rate) {
    for (int i = 0; i < num_leaves_; ++i) {
        leaf_value_[i] = leaf_value_[i] * rate;  // Shrinkage applied here!
    }
    shrinkage_ *= rate;  // Records cumulative shrinkage
}
```

During inference, **do NOT multiply leaf values by shrinkage again**. The `shrinkage` field is purely informational — it records what cumulative shrinkage was applied to this tree.

Example:
- First tree: `shrinkage=1.0` (no scaling)
- Second tree: `shrinkage=0.1` (learning_rate=0.1 was applied)
- Leaf values are already scaled — just sum them directly

---

## Feature Infos Format

`feature_infos` contains metadata for each feature:

- Numerical: `[min:max]` — range of values
- Categorical: `cat1:cat2:cat3:...` — list of category values
- Unused: `none`

Example:
```
feature_infos=[-1.5:2.3] [0:100] 0:1:2:3 none
```
- Feature 0: numerical, range [-1.5, 2.3]
- Feature 1: numerical, range [0, 100]
- Feature 2: categorical, categories {0, 1, 2, 3}
- Feature 3: unused

---

## Parsing Implementation Notes

### Recommended Parsing Strategy

1. Read lines until `Tree=0`
2. Parse header key-value pairs into a HashMap
3. For each tree:
   - Read until next `Tree=` or `end of trees`
   - Parse tree key-value pairs
   - Convert to internal representation
4. Skip remaining sections (feature_importances, parameters, parser)

### Error Handling

Common parsing errors:
- Missing required field
- Array size mismatch
- Invalid number format
- Unexpected end of file

### Performance Considerations

The `tree_sizes` field enables parallel tree parsing. Each tree can be parsed independently since sizes are known upfront.

---

## JSON Format (Alternative)

LightGBM also supports JSON dump via `DumpModel()`. Structure:

```json
{
  "name": "tree",
  "version": "v4",
  "num_class": 1,
  "num_tree_per_iteration": 1,
  "label_index": 0,
  "max_feature_idx": 9,
  "objective": "regression",
  "average_output": false,
  "feature_names": ["f0", "f1", ...],
  "feature_infos": {"f0": {"min_value": -1.5, "max_value": 2.3, "values": []}, ...},
  "tree_info": [
    {
      "tree_index": 0,
      "num_leaves": 31,
      "num_cat": 0,
      "shrinkage": 1.0,
      "tree_structure": {
        "split_index": 0,
        "split_feature": 3,
        "split_gain": 1234.5,
        "threshold": 0.5,
        "decision_type": "<=",
        "default_left": false,
        "missing_type": "None",
        "internal_value": 0.0,
        "internal_weight": 1000.0,
        "internal_count": 1000,
        "left_child": {...},
        "right_child": {...}
      }
    }
  ],
  "feature_importances": {"f0": 100, "f1": 50, ...}
}
```

JSON is more verbose but easier to traverse. Consider supporting both formats.

---

## References

- `src/boosting/gbdt_model_text.cpp` — Model serialization/deserialization
- `src/io/tree.cpp` — Tree serialization (`ToString()`, constructor from string)
- `include/LightGBM/tree.h` — Tree structure and inference logic
- `include/LightGBM/meta.h` — MissingType enum

---

## Comparison: LightGBM vs XGBoost

| Aspect | LightGBM | XGBoost |
|--------|----------|---------|
| Split condition | `≤` (left if ≤ threshold) | `<` (left if < threshold) |
| Leaf indexing | Negative in child arrays | Separate leaf array or embedded |
| Default direction | Bitfield in decision_type | Separate default_left field |
| Missing handling | Zero or NaN modes | NaN only |
| Categorical | Bitset in cat_threshold | Not supported (in standard model) |
| Linear trees | Supported via is_linear | Separate linear booster |
| Format | Text or binary | JSON, text, or binary |

---

## Changelog

- 2024-11-30: Initial research based on LightGBM source code analysis
