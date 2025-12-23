# RFC-0012: Model Compatibility

- **Status**: Implemented
- **Created**: 2024-12-15
- **Updated**: 2025-01-21
- **Scope**: XGBoost and LightGBM model loading

## Summary

Booste-rs provides compatibility loaders for XGBoost and LightGBM models, converting them to native forest and linear model representations for efficient inference.

## Design

### XGBoost Support

**Format**: XGBoost JSON format (≥2.0)

**Loading**: `XgbModel` deserializes directly from JSON via serde. Handles quirks like:
- `base_score` as number, string, array, or bracketed string (`"[1.5E0]"`)
- Boolean fields accepting `true/false`, `0/1`, or string variants

**Boosters Supported**:
- `gbtree` → Tree-based gradient boosting
- `dart` → Dropout trees with per-tree weights
- `gblinear` → Linear model (weights + bias)

**Objectives**: Full support for regression, binary/multiclass classification, ranking, survival, Tweedie, Poisson, etc. Objective name parsed to determine margin-to-probability transform.

**Categorical Features**: Parsed from `split_type`, `categories`, `categories_nodes`, `categories_segments`, `categories_sizes`. Category integer values converted to packed bitsets.

**Base Score Transform**: `prob_to_margin()` applies inverse link based on objective (logit for binary:logistic, log for gamma/tweedie).

### LightGBM Support

**Format**: LightGBM text format (`.txt` files from `save_model()`)

**Parsing**: Line-based key=value format with three sections:
1. Header (metadata, feature info)
2. Trees (one per `Tree=N` block)
3. Footer (ignored)

**Key Differences from XGBoost**:
- Split condition uses `<=` (left) vs XGBoost's `<` → threshold adjusted via `next_up_f32()`
- Leaf indices encoded as negative values (`~child`)
- Decision type is bitfield: bit 0 = categorical, bit 1 = default_left, bits 2-3 = missing type

**Objectives Parsed**: `regression`, `regression_l1`, `binary`, `multiclass`, `multiclassova`

**Categorical Splits**: Bitset stored directly in `cat_threshold`, indexed via `cat_boundaries`.

**Not Supported**: Linear trees (`is_linear = true`)

### Unified Model API

Both loaders convert to native boosters types:

```
XgbModel.to_forest() → Forest<ScalarLeaf>
XgbModel.to_booster() → Booster (Tree | Dart | Linear)

LgbModel.to_forest() → Forest<ScalarLeaf>
```

**Prediction**: `Forest::predict_row(&features) → Vec<f32>` sums tree outputs per group plus base score.

## Key Types

### XGBoost (`src/compat/xgboost/`)

| Type | Description |
|------|-------------|
| `XgbModel` | Top-level parsed model (version, learner) |
| `Learner` | Feature names/types, objective, gradient_booster, model params |
| `GradientBooster` | Enum: `Gbtree`, `Gblinear`, `Dart` |
| `Booster` | Converted enum: `Tree(Forest)`, `Dart{forest, weights}`, `Linear(LinearModel)` |
| `Objective` | Parsed objective with parameters (e.g., `BinaryLogistic`, `MultiSoftprob`) |
| `FeatureType` | `Float`, `Int`, `Categorical`, etc. |
| `ConversionError` | Empty tree, invalid node index, invalid linear weights |

### LightGBM (`src/compat/lightgbm/`)

| Type | Description |
|------|-------------|
| `LgbModel` | Parsed model (header + trees) |
| `LgbHeader` | Metadata: n_classes, feature_names, objective, version |
| `LgbTree` | Tree structure: split_feature, threshold, decision_type, leaf_value |
| `LgbObjective` | Parsed objective: `Regression`, `Binary{sigmoid}`, `Multiclass{n_classes}` |
| `DecisionType` | Decoded bitfield: is_categorical, default_left, missing_type |
| `ParseError` | IO, missing field, invalid value, array mismatch |
| `ConversionError` | Empty tree, invalid child, linear trees unsupported |

### Native Types

| Type | Location | Description |
|------|----------|-------------|
| `Forest<ScalarLeaf>` | `src/repr/gbdt/forest.rs` | Tree collection with group assignments |
| `Tree<ScalarLeaf>` | `src/repr/gbdt/tree.rs` | Single decision tree |
| `LinearModel` | `src/inference/gblinear/model.rs` | Weight matrix + bias |

## Feature Flags

- `xgboost-compat` — Enables XGBoost JSON loading
- `lightgbm-compat` — Enables LightGBM text loading

## Usage Example

```rust
use boosters::compat::xgboost::XgbModel;

// Load XGBoost model
let json: serde_json::Value = serde_json::from_reader(file)?;
let model = XgbModel::from_value(&json)?;

// Convert and predict
match model.to_booster()? {
    Booster::Tree(forest) => {
        let pred = forest.predict_row(&features);
    }
    Booster::Dart { forest, weights } => { /* apply weights */ }
    Booster::Linear(linear) => {
        let pred = linear.predict_row(&features, &base_score);
    }
}
```

```rust
use boosters::compat::lightgbm::LgbModel;

// Load LightGBM model
let model = LgbModel::from_file("model.txt")?;
let forest = model.to_forest()?;
let pred = forest.predict_row(&features);
```

## Changelog

- 2025-01-21: Updated terminology (n_classes) to match codebase conventions
