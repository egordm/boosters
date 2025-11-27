# RFC-0007: Serialization and Model Loading

- **Status**: Accepted
- **Created**: 2024-11-26
- **Related to**: RFC-0001 (Forest), RFC-0002 (Tree)
- **Scope**: Model serialization, external format parsing, format conversion

## Summary

This RFC defines the serialization layer for xgboost-rs:

1. **Native formats**: JSON (human-readable) and binary (efficient) with version stability
2. **External format loaders**: XGBoost JSON, with extensibility for LightGBM, CatBoost
3. **Feature-gated**: External format support is optional (`xgboost-compat` feature)
4. **Stable interchange schema**: Decoupled from internal runtime representation

## Motivation

Model serialization serves multiple purposes:

- **Interop**: Load models trained in Python XGBoost, LightGBM, etc.
- **Persistence**: Save/load native xgboost-rs models across library versions
- **Deployment**: Efficient binary format for production inference

### Design Philosophy

- **Separation of concerns**: Parsing (foreign format → foreign types) vs conversion (foreign → native)
- **Foreign types are temporary**: Exist only for parsing; we don't compute on them
- **Native types are canonical**: All computation uses `NodeForest`/`SoAForest`
- **Stable serialization schema**: Native format survives internal refactoring
- **Feature-gated dependencies**: Heavy serde/macro usage isolated behind features

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    External Formats (feature-gated)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │ XGBoost JSON │  │  LightGBM    │  │   CatBoost   │                   │
│  │  (xgboost-   │  │ (lightgbm-   │  │  (catboost-  │                   │
│  │   compat)    │  │   compat)    │  │   compat)    │                   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                   │
│         │ parse+convert   │                 │                            │
└─────────┼─────────────────┼─────────────────┼────────────────────────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Native Types (core)                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                           Model                                     │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │ │
│  │  │  forest    │  │   meta     │  │ objective  │  │ features   │   │ │
│  │  │ SoAForest  │  │ ModelMeta  │  │ Objective  │  │FeatureInfo │   │ │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│                    Native Serialization (core)                           │
│                           │                                              │
│              ┌────────────┴────────────┐                                │
│              ▼                         ▼                                │
│       ┌────────────┐            ┌────────────┐                          │
│       │   Binary   │            │    JSON    │                          │
│       │  (.xgbrs)  │            │   (.json)  │                          │
│       │  (fast)    │            │ (readable) │                          │
│       └────────────┘            └────────────┘                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module Structure

```text
src/
├── compat/                     # External format support (feature-gated)
│   ├── mod.rs                  # Feature re-exports
│   ├── xgboost_json.rs         # XGBoost JSON: types + parse + convert (all-in-one)
│   ├── lightgbm.rs             # Future: LightGBM text format
│   └── catboost.rs             # Future: CatBoost format
│
├── serialization/              # Native format support (core, always available)
│   ├── mod.rs
│   ├── schema.rs               # Stable interchange types (decoupled from runtime)
│   ├── binary.rs               # Efficient binary format
│   └── json.rs                 # Human-readable JSON format
│
└── model.rs                    # Model, ModelMeta, Objective, FeatureInfo
```

**Note**: Each `compat/` file contains both foreign types AND conversion logic in one place. This avoids Rust's orphan rule issues (can't impl foreign trait for foreign type in third module).

## Design

### Feature Gating

```toml
# Cargo.toml
[features]
default = []
xgboost-compat = ["dep:serde", "dep:serde_json", "dep:serde_with"]
lightgbm-compat = ["dep:serde"]
all-compat = ["xgboost-compat", "lightgbm-compat"]
```

Users who only need native models skip the serde/macro bloat entirely.

### Native Types

```rust
// src/model.rs

/// A trained gradient boosting model (inference-ready)
pub struct Model {
    /// The ensemble (trees, dart, or linear)
    pub booster: Booster,
    
    /// Model metadata
    pub meta: ModelMeta,
    
    /// Feature information
    pub features: FeatureInfo,
    
    /// Objective function (for output transformation)
    pub objective: Objective,
}

/// The ensemble type — concrete types for inference
pub enum Booster {
    /// Standard tree ensemble
    Tree(SoAForest<ScalarLeaf>),
    
    /// DART: trees with per-tree weights applied during inference
    Dart { forest: SoAForest<ScalarLeaf>, weights: Box<[f32]> },
    
    /// Linear model
    Linear(LinearModel),
}

/// Model metadata
pub struct ModelMeta {
    pub num_features: u32,
    pub num_groups: u32,
    pub base_score: Vec<f32>,
    pub source: ModelSource,
}

/// Where the model came from (for diagnostics)
pub enum ModelSource {
    Native { version: u32 },
    XGBoostJson { version: [u32; 3] },
    LightGBM,
    // ...
}
```

**Design note**: `Model` uses concrete `SoAForest` for simplicity and FFI/Python compatibility. Training will use a separate type (e.g., `TrainingModel` with `NodeForest`). If we later need GPU forests or other variants, we can revisit making `Booster` generic or using `Box<dyn Forest>`.

### Feature Metadata — AoS vs SoA **[DECIDED]**

```rust
pub struct FeatureInfo {
    pub features: Vec<Feature>,
}

pub struct Feature {
    pub name: Option<String>,
    pub feature_type: FeatureType,
    pub index: u32,
    // Future: importance, statistics, etc.
}
```

**Rationale**: Feature metadata is accessed infrequently (model inspection, debugging). Colocating fields per-feature is more natural and easier to extend.

### Objective Functions

```rust
pub enum Objective {
    // Regression
    SquaredError,
    AbsoluteError,
    Tweedie { variance_power: f32 },
    Gamma,
    Poisson,
    
    // Binary classification
    BinaryLogistic,
    BinaryLogitRaw,
    
    // Multiclass
    MultiSoftmax { num_class: u32 },
    MultiSoftprob { num_class: u32 },
    
    // Ranking
    RankPairwise,
    RankNdcg,
    
    // Survival
    SurvivalCox,
    SurvivalAft { distribution: AftDistribution },
    
    // ... etc
}
```

**Future**: Training RFC may introduce a trait for custom objectives that compute gradients/hessians. This enum covers inference-time output transformations.

### External Format Loading (XGBoost Example)

```rust
// src/compat/xgboost_json.rs
// All-in-one: foreign types + parsing + conversion

#[cfg(feature = "xgboost-compat")]

// --- Foreign types (mirror XGBoost JSON schema exactly) ---

#[derive(Deserialize)]
pub(crate) struct XgbModel { /* ... */ }

#[derive(Deserialize)]  
pub(crate) struct XgbTree { /* ... */ }

// --- Public API ---

pub fn load<P: AsRef<Path>>(path: P) -> Result<Model, LoadError> {
    let xgb: XgbModel = serde_json::from_reader(File::open(path)?)?;
    convert_to_native(xgb)
}

pub fn load_str(json: &str) -> Result<Model, LoadError> {
    let xgb: XgbModel = serde_json::from_str(json)?;
    convert_to_native(xgb)
}

// --- Conversion (private, same module avoids orphan rule) ---

fn convert_to_native(xgb: XgbModel) -> Result<Model, LoadError> {
    // Convert trees, metadata, objective...
    // Details elided
}
```

### Native Serialization — Stable Schema

**Problem**: Internal runtime types (`SoAForest`, `SoATreeStorage`) may change between versions. We need saved models to remain loadable.

**Solution**: Define a stable **interchange schema** separate from runtime types.

```rust
// src/serialization/schema.rs

/// Stable schema version (bump on breaking changes)
pub const SCHEMA_VERSION: u32 = 1;

/// Interchange representation — stable across library versions
/// This is what gets serialized, not the runtime types directly.
pub struct ModelSchema {
    pub schema_version: u32,
    pub booster: BoosterSchema,
    pub meta: MetaSchema,
    pub features: Vec<FeatureSchema>,
    pub objective: ObjectiveSchema,
}

pub enum BoosterSchema {
    Tree(TreeEnsembleSchema),
    Dart { trees: TreeEnsembleSchema, weights: Vec<f32> },
    Linear { weights: Vec<f32>, num_features: u32, num_groups: u32 },
}

pub struct TreeEnsembleSchema {
    pub trees: Vec<TreeSchema>,
    pub tree_groups: Vec<u32>,
}

pub struct TreeSchema {
    pub split_indices: Vec<u32>,
    pub split_values: Vec<f32>,
    pub left_children: Vec<u32>,
    pub right_children: Vec<u32>,
    pub default_left: Vec<bool>,
    pub leaf_values: Vec<f32>,
    pub is_leaf: Vec<bool>,
    // Categorical data if present
    pub categorical: Option<CategoricalSchema>,
}
```

**Conversion flow**:

```text
Runtime types ←→ Schema types ←→ JSON/Binary bytes
     ↑                ↑                ↑
  (fast,          (stable,         (on disk)
   internal)       versioned)
```

### Binary Format

```rust
// src/serialization/binary.rs

const MAGIC: &[u8; 5] = b"XGBRS";

pub fn save<W: Write>(model: &Model, writer: W) -> Result<()> {
    let schema = ModelSchema::from_runtime(model);
    // Write: magic, version, length-prefixed binary data
    // Could use bincode, postcard, or custom encoding
}

pub fn load<R: Read>(reader: R) -> Result<Model> {
    // Read magic, check version, deserialize schema, convert to runtime
}
```

### JSON Format

```rust
// src/serialization/json.rs

pub fn save<W: Write>(model: &Model, writer: W) -> Result<()> {
    let schema = ModelSchema::from_runtime(model);
    serde_json::to_writer_pretty(writer, &schema)?;
    Ok(())
}

pub fn load<R: Read>(reader: R) -> Result<Model> {
    let schema: ModelSchema = serde_json::from_reader(reader)?;
    schema.into_runtime()
}
```

## Design Decisions

### DD-1: Single-File per External Format **[DECIDED]**

**Decision**: Each external format is one file (e.g., `xgboost_json.rs`) containing foreign types AND conversion.

**Rationale**: Avoids Rust orphan rule issues. Can't implement `TryFrom` in a separate module from the types.

### DD-2: Feature-Gated External Formats **[DECIDED]**

**Decision**: External format support behind cargo features (`xgboost-compat`, etc.).

**Rationale**: Users doing inference-only with native models skip serde bloat.

### DD-3: Stable Interchange Schema **[DECIDED]**

**Decision**: Serialize via stable schema types, not runtime types directly.

**Rationale**: Internal refactoring won't break saved models. Schema version enables migrations.

### DD-4: DART as Separate Booster Variant **[DECIDED]**

**Decision**: `DartForest` wraps forest + weights, not a side field.

**Rationale**: Inference uses weights. Must be part of the type to ensure correct predict path.

### DD-5: Feature Metadata Layout **[DECIDED]**

**Decision**: AoS (`Vec<Feature>`) — one struct per feature.

**Rationale**: Metadata accessed infrequently; colocating fields is natural and extensible.

### DD-6: Binary Format Encoding **[DECIDED]**

**Decision**: Use `bincode` crate.

**Rationale**:

- Well-maintained, widely used
- Simple API, low effort
- Robust — avoids hand-rolled encoding bugs
- Good performance

**Alternatives rejected**:

- Custom encoding: High effort, error-prone
- `postcard`: Good for no_std, but we don't need that constraint

### DD-7: Concrete Types for Inference Model **[DECIDED]**

**Decision**: `Model` and `Booster` use concrete `SoAForest<ScalarLeaf>`, not generics.

**Rationale**:

- Simpler API — no generic parameter to specify
- FFI/Python friendly — concrete types at boundary
- No performance cost — hot path is inside forest, not at Model level
- Linear booster doesn't need a forest type parameter

**Future consideration**: If GPU forests or training integration require polymorphism, revisit with `Box<dyn Forest>` or generics. For now, training uses a separate `TrainingModel` type.

## Future Extensions

- **LightGBM/CatBoost loaders**: Same pattern as XGBoost
- **ONNX export**: Convert to ONNX tree ensemble
- **Custom objectives**: Training RFC may add trait for gradient/hessian computation

## Integration with Other RFCs

| RFC | Integration |
|-----|-------------|
| RFC-0001 | Conversion creates `NodeForest`, calls `freeze()` |
| RFC-0002 | Tree schema mirrors `NodeTree` structure |
| RFC-0003 | `Model::predict()` uses `Predictor` |
| RFC-0004 | `predict()` takes `DataMatrix` |

## References

- XGBoost JSON schema: `doc/model.schema`
- Existing parser: `src/loaders/xgboost/format.rs`
- [bincode crate](https://docs.rs/bincode)
- [postcard crate](https://docs.rs/postcard)
