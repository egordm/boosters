# RFC-0021: Native Storage Format

- **Status**: Accepted
- **Created**: 2025-06-12
- **Updated**: 2025-12-19
- **Depends on**: RFC-0002 (Forest/Tree), RFC-0014 (GBLinear), RFC-0015 (Linear Trees)
- **Scope**: Model serialization, native storage format, Python conversion utilities

## Summary

Define a native storage format for booste-rs that supports all model types (GBDT, GBLinear, DART) including advanced features (linear leaves, categorical splits). Replace Rust-based XGBoost/LightGBM parsers with Python conversion utilities to reduce maintenance burden.

## Motivation

### Current State

booste-rs has Rust parsers for XGBoost JSON and LightGBM text formats (RFC-0012). Models trained within booste-rs cannot be persisted.

### Problems

1. **No save/load for native models**: Training produces models that cannot be saved
2. **Parser maintenance burden**: XGBoost and LightGBM formats evolve; keeping Rust parsers current is ongoing work
3. **Feature gaps in external formats**: No single external format supports all our features (see [storage format research](../research/storage-formats.md))
4. **Language mismatch**: Rust is not ideal for parsing text formats with quirks; Python is more suitable

### Proposed Solution

1. Define a native `.bstr` format that captures all booste-rs capabilities
2. Remove Rust-based XGBoost/LightGBM parsers from the core library
3. Provide Python conversion utilities: `xgboost_to_bstr()`, `lightgbm_to_bstr()`
4. Python converters use XGBoost/LightGBM's own APIs for reliable parsing

### Benefits

| Aspect | Current (Rust parsers) | Proposed (Python converters) |
| ------ | ---------------------- | ---------------------------- |
| Maintenance | High: track format changes | Low: use native library APIs |
| Correctness | Risk: reimplementing parsing | High: use official APIs |
| Dependencies | None (pure Rust) | Optional Python packages |
| Conversion speed | Fast | One-time cost, acceptable |
| Runtime loading | Direct from external format | Load `.bstr` only (fast) |

## Design

### Format Requirements

The native format must support:

1. **GBDT models** with categorical splits
2. **Linear leaves** (per-leaf coefficients, intercepts, feature indices)
3. **GBLinear models** (weight matrix, bias vector)
4. **DART models** (per-tree dropout weights)
5. **Metadata**: feature names, types, objective, base scores
6. **Versioning**: forward/backward compatibility
7. **Integrity**: corruption detection (checksum)

### Format Choice: Binary vs JSON

#### Option A: Pure Binary (Postcard + zstd)

**Pros**: Compact, fast to load, simple implementation  
**Cons**: Not human-readable, debugging requires tooling

#### Option B: Pure JSON

**Pros**: Human-readable, easy debugging, diffable  
**Cons**: Large files (5-10x), slow parsing, verbose

#### Decision: Binary Primary with JSON Debug Export

Use **binary** as the production format with a separate JSON export for debugging:

- `.bstr` — Binary format (production, default)
- `model.to_json()` — JSON string for inspection (not for loading back)

#### Format Header

The header is exactly 32 bytes with explicit offsets:

```rust
/// Magic number: "BSTR" (0x42535452)
const MAGIC: [u8; 4] = *b"BSTR";

/// Header layout (32 bytes, packed):
/// 
/// Offset  Size  Field
/// ------  ----  -----
/// 0       4     magic ("BSTR")
/// 4       2     version_major
/// 6       2     version_minor
/// 8       1     model_type
/// 9       1     flags
/// 10      6     reserved (zeros)
/// 16      8     payload_size
/// 24      4     checksum (CRC32)
/// 28      4     padding (zeros)
/// ------  ----
/// 32 bytes total

#[repr(C, packed)]
pub struct FormatHeader {
    pub magic: [u8; 4],
    pub version_major: u16,
    pub version_minor: u16,
    pub model_type: u8,
    pub flags: u8,
    pub reserved: [u8; 6],
    pub payload_size: u64,
    pub checksum: u32,
    pub padding: [u8; 4],
}

/// Model type discriminant
pub mod ModelType {
    pub const GBDT: u8 = 0;
    pub const DART: u8 = 1;
    pub const GBLINEAR: u8 = 2;
}

/// Feature flags (bitfield, 8 bits)
pub mod Flags {
    pub const COMPRESSED: u8 = 1 << 0;
    pub const HAS_CATEGORICAL: u8 = 1 << 1;
    pub const HAS_LINEAR_LEAVES: u8 = 1 << 2;
    pub const DOUBLE_PRECISION: u8 = 1 << 3;
    // Bits 4-7 reserved for future use
}

/// Minimum size threshold for compression (32KB)
const COMPRESSION_THRESHOLD: usize = 32 * 1024;
```

#### Payload Envelope

The payload uses a version-tagged enum for forward compatibility:

```rust
/// Top-level payload with version tag
#[derive(Serialize, Deserialize)]
pub enum Payload {
    /// Version 1 payload format
    V1(PayloadV1),
    // V2(PayloadV2) added when major changes needed
}

/// V1 payload structure
#[derive(Serialize, Deserialize)]
pub struct PayloadV1 {
    pub metadata: Metadata,
    pub model: ModelData,
}
```

This ensures:
- v1.0 readers encountering V2 get a clear deserialization error
- Minor version additions use `#[serde(default)]` on optional fields
- Major version changes add new enum variants

#### GBDT Payload

```rust
/// Serialized GBDT model
pub struct GbdtPayload {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Forest structure
    pub forest: ForestPayload,
    /// Optional: Categorical split data
    pub categories: Option<CategoriesPayload>,
    /// Optional: Linear leaf coefficients
    pub linear_leaves: Option<LinearLeavesPayload>,
}

pub struct ModelMetadata {
    /// Number of features
    pub num_features: u32,
    /// Number of output groups (1 for regression, num_classes for multiclass)
    pub num_groups: u32,
    /// Base score per group
    pub base_scores: Vec<f32>,
    /// Objective function name (for post-processing)
    pub objective: String,
    /// Feature names (optional)
    pub feature_names: Option<Vec<String>>,
    /// Feature types (optional)
    pub feature_types: Option<Vec<FeatureType>>,
    /// Arbitrary key-value attributes
    pub attributes: HashMap<String, String>,
}

pub struct ForestPayload {
    /// Number of trees
    pub num_trees: u32,
    /// Tree-to-group assignment
    pub tree_groups: Vec<u32>,
    /// Tree structures (SoA layout)
    pub trees: Vec<TreePayload>,
}

pub struct TreePayload {
    /// Number of nodes
    pub num_nodes: u32,
    /// Split feature indices
    pub split_features: Vec<i32>,
    /// Split thresholds
    pub thresholds: Vec<f32>,
    /// Left child indices (negative = leaf)
    pub left_children: Vec<i32>,
    /// Right child indices (negative = leaf)
    pub right_children: Vec<i32>,
    /// Default direction flags
    pub default_left: BitVec,
    /// Categorical split flags
    pub is_categorical: BitVec,
    /// Leaf values (indexed by ~leaf_idx)
    pub leaf_values: Vec<f32>,
    /// Node-to-category-range mapping (if categorical)
    pub category_offsets: Option<Vec<u32>>,
}
```

#### Linear Leaves Extension

```rust
/// Linear leaf coefficients storage (RFC-0015)
pub struct LinearLeavesPayload {
    /// Per-tree: which leaves have linear terms
    pub tree_leaf_masks: Vec<BitVec>,
    /// Packed coefficients: (tree, leaf) -> (features, coefficients, intercept)
    pub coefficients: LinearCoefficientsStorage,
}

pub struct LinearCoefficientsStorage {
    /// Offsets into coefficient arrays per (tree, leaf)
    pub offsets: Vec<u32>,
    /// Feature indices for each linear leaf
    pub feature_indices: Vec<u16>,
    /// Coefficient values
    pub coefficients: Vec<f32>,
    /// Intercepts (one per linear leaf)
    pub intercepts: Vec<f32>,
}
```

#### GBLinear Payload

```rust
/// Serialized GBLinear model (RFC-0014)
pub struct GbLinearPayload {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Weight matrix: [num_features + 1, num_groups] in row-major
    /// Last row is bias
    pub weights: Vec<f32>,
    /// Number of boosting rounds completed
    pub num_boosted_rounds: u32,
}
```

#### DART Extension

```rust
/// DART-specific data
pub struct DartPayload {
    /// Base GBDT model
    pub gbdt: GbdtPayload,
    /// Per-tree weights (for DART dropout)
    pub tree_weights: Vec<f32>,
}
```

### Serialization Codec

```rust
/// Format codec for reading/writing models
pub trait ModelCodec {
    /// Serialize model to bytes
    fn serialize(&self, model: &impl Model) -> Result<Vec<u8>, SerializeError>;
    
    /// Deserialize model from bytes
    fn deserialize(bytes: &[u8]) -> Result<Box<dyn Model>, DeserializeError>;
    
    /// Write model to a writer
    fn write_to(&self, model: &impl Model, writer: &mut impl Write) 
        -> Result<(), SerializeError>;
    
    /// Read model from a reader
    fn read_from(reader: &mut impl Read) -> Result<Box<dyn Model>, DeserializeError>;
}

/// Native booste-rs format codec
pub struct NativeCodec {
    /// Whether to compress payload with zstd
    pub compress: bool,
    /// Compression level (if enabled)
    pub compression_level: i32,
}

impl Default for NativeCodec {
    fn default() -> Self {
        Self {
            compress: true,
            compression_level: 3,
        }
    }
}
```

### Python Integration (RFC-0019)

```python
class GBDTBooster:
    def save(self, path: str, *, compress: bool = True) -> None:
        """Save model to .bstr file.
        
        Args:
            path: File path (should end with .bstr)
            compress: Enable zstd compression (default True)
        """
        ...
    
    @classmethod
    def load(cls, path: str) -> "GBDTBooster":
        """Load model from .bstr file.
        
        Raises:
            FileNotFoundError: If path does not exist
            ValueError: If file is not a valid booste-rs model
            ValueError: If model requires a newer booste-rs version
            IOError: If file is corrupted (checksum mismatch)
        """
        ...
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes (native format)."""
        ...
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "GBDTBooster":
        """Deserialize from bytes."""
        ...
    
    def __reduce__(self) -> tuple:
        """Pickle support via to_bytes/from_bytes."""
        return (self.__class__.from_bytes, (self.to_bytes(),))
    
    def to_json(self) -> str:
        """Export as JSON string for debugging/inspection.
        
        Note: This is for debugging only. Use save() for persistence.
        """
        ...
```

### Python Conversion Utilities

Replace Rust parsers with Python converters that use library **introspection APIs** (not format parsing):

```python
# boosters/convert.py

def xgboost_to_bstr(
    model: "xgboost.Booster",
    path: str,
    *,
    include_feature_names: bool = True,
) -> None:
    """Convert XGBoost model to boosters .bstr format.
    
    Uses XGBoost's introspection APIs to extract model structure:
    - model.trees_to_dataframe() for tree structure
    - model.get_config() for objective, base_score
    - model.feature_names for feature names
    
    This avoids parsing XGBoost's serialization format directly.
    
    Args:
        model: Trained XGBoost Booster object
        path: Output .bstr file path
        include_feature_names: Include feature names in metadata
    
    Example:
        import xgboost as xgb
        from boosters.convert import xgboost_to_bstr
        
        # Train with XGBoost
        xgb_model = xgb.train(params, dtrain)
        
        # Convert to boosters format (one-time)
        xgboost_to_bstr(xgb_model, "model.bstr")
        
        # Load with boosters (fast, for inference)
        import boosters
        model = boosters.GBDTBooster.load("model.bstr")
    
    Supports:
        - gbtree booster (GBDT)
        - dart booster (with tree weights)
        - gblinear booster (linear model)
        - Categorical features
    """
    ...


def lightgbm_to_bstr(
    model: "lightgbm.Booster",
    path: str,
    *,
    include_feature_names: bool = True,
) -> None:
    """Convert LightGBM model to boosters .bstr format.
    
    Uses LightGBM's introspection APIs:
    - model.trees_to_dataframe() for standard tree structure
    - model.dump_model() for linear tree coefficients
    - model.params for configuration
    
    This avoids parsing LightGBM's text format directly.
    
    Args:
        model: Trained LightGBM Booster object
        path: Output .bstr file path
        include_feature_names: Include feature names in metadata
        
    Example:
        import lightgbm as lgb
        from boosters.convert import lightgbm_to_bstr
        
        # Train with LightGBM
        lgb_model = lgb.train(params, train_set)
        
        # Convert to boosters format (one-time)
        lightgbm_to_bstr(lgb_model, "model.bstr")
        
        # Load with boosters (fast, for inference)
        import boosters
        model = boosters.GBDTBooster.load("model.bstr")
    
    Supports:
        - Standard GBDT
        - Linear trees (via dump_model() for coefficients)
        - Categorical features
    """
    ...


def sklearn_to_bstr(
    model: "sklearn.ensemble.BaseEnsemble",
    path: str,
) -> None:
    """Convert scikit-learn ensemble to boosters .bstr format.
    
    Uses direct attribute access on sklearn tree objects.
    
    Supports:
        - GradientBoostingClassifier
        - GradientBoostingRegressor
        - RandomForestClassifier
        - RandomForestRegressor
        - HistGradientBoostingClassifier
        - HistGradientBoostingRegressor
    """
    ...
```

### Migration Path from Rust Parsers

```python
# BEFORE (Rust parsers - being deprecated):
# The current approach requires maintaining Rust code for each format
import boosters
model = boosters.compat.load_xgboost("model.json")  # Rust parsing

# AFTER (Python converters - new approach):
# Use official library APIs, convert once, load fast thereafter
import xgboost as xgb
from boosters.convert import xgboost_to_bstr
import boosters

# One-time conversion using XGBoost's own API
xgb_model = xgb.Booster()
xgb_model.load_model("model.json")
xgboost_to_bstr(xgb_model, "model.bstr")

# Fast native loading for inference
model = boosters.GBDTBooster.load("model.bstr")
predictions = model.predict(X)
```

### Deprecation Plan for Rust Parsers

| Version | Status |
| ------- | ------ |
| v0.x | Add native format and Python converters |
| v0.x+1 | Deprecate `XgbModel::from_file()`, `LgbModel::from_file()` |
| v1.0 | Remove Rust parsers from core; Python converters are primary path |

### Pure-Rust Workflows

For users who need pure-Rust (no Python) workflows:

1. **Recommended**: Pre-convert models using Python during build/deployment
2. **Alternative**: Use models trained natively with booste-rs
3. **Future option**: Community-maintained `booste-rs-compat` crate

The `booste-rs-compat` crate could provide XGBoost/LightGBM parsers with:
- Feature-gated behind `xgboost` and `lightgbm` flags
- Lower support commitment (community-maintained)
- "Best effort" compatibility with format changes

This keeps the core library lean while providing an escape hatch for pure-Rust use cases.

### Version Compatibility

```text
┌─────────────────────────────────────────────────────────────┐
│                   Version Compatibility                      │
├─────────────────────────────────────────────────────────────┤
│  v1.x can read:  v1.0, v1.1, v1.2, ...                      │
│  v1.x cannot read: v2.x (major version bump)                │
│                                                              │
│  Forward compatibility: Reject with clear version message   │
│  Backward compatibility: Newer readers handle old formats   │
└─────────────────────────────────────────────────────────────┘
```

**Version Rules**:

- Major version bump: Breaking changes to existing fields
- Minor version bump: New optional fields, new model types
- Patch versions are not encoded (library patches don't affect format)

**Error Messages**:

| Condition | Message |
| --------- | ------- |
| Wrong magic | "Not a booste-rs model file" |
| Major version too high | "Model requires booste-rs 2.x, you have 1.x" |
| Minor version too high | "Model requires booste-rs 1.3+, you have 1.0" |
| Unknown ModelType | "Unknown model type 5; upgrade booste-rs" |
| Checksum mismatch | "File corrupted: checksum verification failed" |
| Truncated file | "File truncated: expected {n} bytes, got {m}" |

## Testing Strategy

### Format Version Compatibility Tests

Maintain a corpus of serialized models for each format version:

```text
tests/format-corpus/
├── v1.0/
│   ├── gbdt-simple.bstr
│   ├── gbdt-categorical.bstr
│   ├── gbdt-linear-leaves.bstr
│   ├── gblinear.bstr
│   └── dart.bstr
├── v1.1/
│   └── (added when v1.1 ships)
```

**Test matrix**:

- Current version can read all older versions in corpus
- Checksums are verified on load
- Predictions match expected values

### Corruption Detection Tests

```rust
#[test]
fn test_truncated_file() {
    let bytes = model.to_bytes();
    let truncated = &bytes[..bytes.len() - 100];
    assert!(matches!(
        Model::from_bytes(truncated),
        Err(DeserializeError::Truncated { .. })
    ));
}

#[test]
fn test_corrupted_payload() {
    let mut bytes = model.to_bytes();
    bytes[50] ^= 0xFF;  // Flip bits in payload
    assert!(matches!(
        Model::from_bytes(&bytes),
        Err(DeserializeError::ChecksumMismatch)
    ));
}

#[test]
fn test_wrong_magic() {
    let mut bytes = model.to_bytes();
    bytes[0..4].copy_from_slice(b"XXXX");
    assert!(matches!(
        Model::from_bytes(&bytes),
        Err(DeserializeError::NotAModel)
    ));
}
```

### Round-Trip Tests

```rust
#[test]
fn test_round_trip_preserves_predictions() {
    let model = train_model(&dataset);
    let predictions_before = model.predict(&test_data);
    
    let bytes = model.to_bytes();
    let loaded = Model::from_bytes(&bytes).unwrap();
    let predictions_after = loaded.predict(&test_data);
    
    assert_predictions_eq(&predictions_before, &predictions_after);
}
```

### Python Converter Tests

```python
def test_xgboost_converter_roundtrip():
    """Verify XGBoost converter produces correct predictions."""
    import xgboost as xgb
    from boosters.convert import xgboost_to_bstr
    import boosters
    
    # Train XGBoost model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    xgb_model = xgb.train({"objective": "reg:squarederror"}, dtrain)
    
    # Convert to boosters format
    xgboost_to_bstr(xgb_model, "model.bstr")
    
    # Load and compare predictions
    model = boosters.GBDTBooster.load("model.bstr")
    
    xgb_preds = xgb_model.predict(xgb.DMatrix(X_test))
    our_preds = model.predict(X_test)
    
    np.testing.assert_allclose(xgb_preds, our_preds, rtol=1e-6)


def test_lightgbm_converter_with_linear_trees():
    """Verify LightGBM linear trees are converted correctly."""
    import lightgbm as lgb
    from boosters.convert import lightgbm_to_bstr
    import boosters
    
    # Train LightGBM with linear trees
    params = {"objective": "regression", "linear_tree": True}
    train_set = lgb.Dataset(X_train, label=y_train)
    lgb_model = lgb.train(params, train_set)
    
    # Convert and verify
    lightgbm_to_bstr(lgb_model, "model.bstr")
    model = boosters.GBDTBooster.load("model.bstr")
    
    lgb_preds = lgb_model.predict(X_test)
    our_preds = model.predict(X_test)
    
    np.testing.assert_allclose(lgb_preds, our_preds, rtol=1e-6)
```

## Design Decisions

### DD-1: Python Converters over Rust Parsers

**Context**: How to support loading XGBoost/LightGBM models?

**Options considered**:

1. **Rust parsers** — Maintain Rust code for each external format
2. **Python converters** — Use official library APIs in Python, convert to native format
3. **Both** — Keep Rust parsers for pure-Rust use cases

**Decision**: Python converters (#2)

**Rationale**:

- XGBoost/LightGBM APIs handle their own format quirks correctly
- Python is better suited for parsing complex text formats
- One-time conversion cost is acceptable for inference workloads
- Removes significant maintenance burden from Rust codebase

**Consequences**:

- Users must have XGBoost/LightGBM Python packages to convert
- Pure-Rust workflows cannot load external formats (must pre-convert)
- But: simpler codebase, correct parsing guaranteed

### DD-2: Binary Format with Postcard

**Context**: What serialization library/format to use?

**Options considered**:

1. **JSON** — Human readable, large files
2. **MessagePack** — Compact, but no Rust ecosystem standard
3. **Bincode** — Fast, compact, Rust-native
4. **Postcard** — Smaller than bincode, no_std compatible
5. **FlatBuffers/Cap'n Proto** — Zero-copy, but complex API

**Decision**: Postcard with optional zstd compression

**Rationale**:

- Postcard is more compact than bincode
- no_std compatible (future embedded use cases)
- Simple serde derive-based API
- zstd provides excellent compression for large models

**Consequences**:

- Dependency on `postcard` and optionally `zstd`
- Not human-readable without tooling (hence JSON debug export)

### DD-3: Explicit Version Header

**Context**: How to handle format evolution?

**Options considered**:

1. **Implicit versioning** — Rely on serde's tagged enums
2. **Explicit header** — Fixed header with version numbers
3. **Schema evolution** — Protobuf-style field numbers

**Decision**: Explicit header with magic + semver

**Rationale**:

- Magic number enables quick format detection
- Semver-style versioning is familiar
- Can detect incompatibility before deserialization
- Fixed 32-byte header is simple and efficient

**Consequences**:

- Must maintain version compatibility guarantees
- But: clear contract for users, fast rejection of incompatible files

### DD-4: Compression with Auto-Threshold

**Context**: Should models be compressed?

**Options considered**:

1. **Always compressed** — Maximum space efficiency
2. **Never compressed** — Maximum read speed
3. **Configurable with threshold** — Compress large models only

**Decision**: On by default, auto-skip for small models (<32KB)

**Rationale**:

- Small models don't benefit much from compression
- Large models (100+ trees) benefit significantly (3-5x reduction)
- zstd decompression is fast enough that overhead is minimal
- Auto-threshold removes need for user tuning

**Consequences**:

- Slight implementation complexity
- But: optimal tradeoff without user configuration

## Integration

| Component | Integration Point | Notes |
| --------- | ----------------- | ----- |
| RFC-0002 (Forest) | `Forest::save/load` | Primary serialization target |
| RFC-0014 (GBLinear) | `GbLinearPayload` | Full linear model serialization |
| RFC-0015 (Linear Trees) | `LinearLeavesPayload` | Per-leaf coefficient storage |
| RFC-0019 (Python) | `save/load/pickle` | Python API integration |
| RFC-0012 (Compat) | Deprecation | Rust parsers replaced by Python converters |

## File Extension

| Extension | Description |
| --------- | ----------- |
| `.bstr` | Native booste-rs binary format |

## Open Questions

1. **Compression algorithm**: zstd (better ratio) vs lz4 (faster)?
   - Proposal: zstd level 3 (good balance)

2. **Float precision**: Always f32, or support f64?
   - Proposal: f32 default, f64 flag for high-precision models

3. **Streaming support**: Load trees one-by-one for huge models?
   - Proposal: Defer to future version

## Future Work

- [ ] Treelite v4 export for ONNX/TL2cgen interop
- [ ] ONNX export for deployment
- [ ] Memory-mapped large model support
- [ ] Encrypted model storage (for proprietary models)

## References

- [Storage Format Research](../research/storage-formats.md) — Treelite/XGBoost/LightGBM evaluation
- [Postcard Crate](https://docs.rs/postcard)
- [RFC-0002 (Forest/Tree)](0002-forest-and-tree-structures.md)
- [RFC-0014 (GBLinear)](0014-gblinear.md)
- [RFC-0015 (Linear Trees)](0015-linear-trees.md)

## Changelog

- 2025-06-12: Initial draft with Treelite evaluation
- 2025-12-19: Refocused on native format only
- 2025-12-19: Moved Treelite research to docs/research/storage-formats.md
- 2025-12-19: Added Python converter proposal to replace Rust parsers
- 2025-12-19: Simplified design decisions to match new scope
- 2025-12-19: Round 1-2: Clarified Python converters use introspection APIs
- 2025-12-19: Round 3: Fixed header to exactly 32 bytes with explicit offsets
- 2025-12-19: Round 4: Added pure-Rust workflow documentation and compat crate option
- 2025-12-19: Round 5: Added payload envelope with version-tagged enum for forward compatibility
- 2025-12-19: Round 6: Final review, ready for acceptance
