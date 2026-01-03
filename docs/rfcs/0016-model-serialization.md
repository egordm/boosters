# RFC-0016: Model Serialization

**Status**: Implemented  
**Created**: 2026-01-02  
**Updated**: 2026-01-03  
**Author**: Team

## Implementation Tracking

Implementation work is tracked in [docs/backlogs/0016-model-serialization.md](../backlogs/0016-model-serialization.md).

## Summary

- Define a native `.bstr` format for serializing and deserializing boosters models.
- Support all model types: GBDT, GBLinear, and future variants (DART, linear trees).
- Provide both human-readable JSON and binary compressed representations.
- Enable versioning and backward compatibility for schema evolution.
- Allow serialization of model subcomponents (trees, forests, leaves) for inspection/visualization.
- Provide a Python schema mirror for `.bstr.json` to enable native parsing and explainability tooling.
- Remove the Rust compat layer (`crates/boosters/src/compat`); model conversion moves to Python utilities.

## Motivation

### Current State

boosters currently has no native serialization format. Models trained with boosters cannot be saved and loaded without converting to/from XGBoost or LightGBM formats via the compat layer. This has several problems:

1. **Lossy conversion**: Not all boosters features map cleanly to XGBoost/LightGBM (e.g., linear leaves, multi-output trees).
2. **Maintenance burden**: The compat layer in `crates/boosters/src/compat` (XGBoost JSON, LightGBM text) is complex and requires ongoing updates for format changes.
3. **Testing friction**: Integration tests use XGBoost JSON models, requiring the compat layer even for internal tests.
4. **No model persistence**: Users training models with boosters have no way to save them for production deployment.

### Goals

1. **Native persistence**: Save and load boosters models directly without external format dependencies.
2. **Future-proof**: Version the format so older models can be loaded by newer library versions.
3. **Inspection**: Allow tools (Python, visualization) to read model structure for plotting, debugging, and analysis.
4. **Simplify codebase**: Remove compat layer from Rust crate; provide optional Python conversion utilities.

## Non-Goals

- Define a universal ML model interchange format (not ONNX, PMML).
- Partial model loading (e.g., loading a subset of trees without reading all payload).
- Support loading arbitrary XGBoost/LightGBM models in Rust (this moves to Python).
- Provide format converters in languages other than Python.
- Provide a `bstr` CLI tool for format inspection/conversion (future work).
- Memory-mapped loading for very large models (future optimization).
- Pure-Python parsing of binary `.bstr` (MessagePack / zstd) without the Rust extension module.

## Design

### Format Overview

The `.bstr` format is a container that can hold:

1. **Header**: Magic bytes, schema version, format variant, model type.
2. **Metadata**: Model type, feature info, task kind, training config.
3. **Payload**: Model-specific data (forest, linear model, etc.).
4. **Trailer (binary only)**: Payload length and checksum.

### Format Variants

| Variant | Extension | Use Case |
| ------- | --------- | -------- |
| JSON | `.bstr.json` | Human-readable, debugging, inspection |
| Binary | `.bstr` | Production, size/speed optimized |

The binary format uses MessagePack with optional zstd compression.

### Binary Format Specification

The binary `.bstr` format is designed to support one-pass streaming writes into any `std::io::Write` without requiring `Seek`.

It uses a fixed-size header and fixed-size trailer:

1. Write the 32-byte header
2. Stream the payload (MessagePack, optionally zstd-compressed)
3. Write the 16-byte trailer

The checksum and payload length live in the trailer so they can be computed while streaming.

#### Header (32 bytes)

| Offset | Size | Field | Description |
| ------ | ---- | ----- | ----------- |
| 0 | 4 | Magic | ASCII `BSTR` (0x42 0x53 0x54 0x52) |
| 4 | 4 | Schema version | Little-endian u32 |
| 8 | 1 | Format | 0x01 = MessagePack, 0x02 = MessagePack+zstd |
| 9 | 1 | Model type | See `ModelTypeId` |
| 10 | 22 | Reserved | Zero bytes, reserved for future header fields |

#### Payload (N bytes)

The payload is MessagePack bytes (optionally zstd-compressed), streamed directly to the writer.

#### Trailer (16 bytes)

The trailer is appended at end-of-stream:

| Offset (from end) | Size | Field | Description |
| ----------------- | ---- | ----- | ----------- |
| -16 | 8 | Payload length | Little-endian u64 (number of payload bytes written) |
| -8 | 4 | Checksum | CRC32C (Castagnoli) of payload bytes, little-endian |
| -4 | 4 | Reserved | Zero bytes, reserved for future trailer fields |

**Header size**: The header is padded to 32 bytes for alignment and future extensibility. Reserved bytes must be zero; non-zero values in reserved bytes are ignored for forward compatibility.

**Endianness**: All multi-byte integers are little-endian. This includes integers within the MessagePack payload (MessagePack uses big-endian by spec, but our schema uses raw bytes for bitsets and arrays which must be little-endian).

**CRC32C**: We use CRC32C (Castagnoli polynomial) for checksum, which has hardware acceleration on modern CPUs via SSE4.2/ARMv8.

**Compression**: When format byte is 0x02, the payload is zstd-compressed. Decompression yields MessagePack bytes. Default compression level is 3 (fast with good ratio). For reference:

- Level 1: ~300 MB/s compression, lower ratio
- Level 3: ~150 MB/s compression, good balance (default)
- Level 9+: <50 MB/s compression, diminishing returns for model data

**Format choice guidance**:

- Use binary (`.bstr`) for production: 10-20x smaller than JSON, faster to load
- Use JSON (`.bstr.json`) for debugging, inspection, or manual editing

**Streaming decode note**: For non-seekable readers, decoding needs to buffer the last 16 bytes to separate payload from trailer. This is implemented with a small ring buffer and does not require buffering the full payload.

### Schema Versioning

Each `.bstr` file includes a schema version number:

```text
schema_version: u32  // e.g., 1
```

**Compatibility rules**:

- **Backward compatible**: Newer library versions can always load older schema versions.
- **Forward compatible (best-effort)**: Unknown fields are ignored; unknown enum variants fail gracefully with clear error messages.
- **Breaking changes**: Increment schema version; provide migration code for previous versions.

### Envelope Structure

```rust
struct BstrHeader {
    /// Magic bytes: "BSTR"
    magic: [u8; 4],
    /// Schema version (monotonically increasing)
    schema_version: u32,
    /// Format variant: 0x01 = MessagePack, 0x02 = MessagePack+zstd
    format: u8,
    /// Model type discriminant
    model_type: ModelTypeId,
    /// Reserved header bytes (must be zero on write)
    reserved: [u8; 22],
}

struct BstrTrailer {
    /// Payload length in bytes (number of payload bytes written)
    payload_len: u64,
    /// Checksum (CRC32C of payload bytes)
    checksum: u32,
    /// Reserved trailer bytes (must be zero on write)
    reserved: [u8; 4],
}
```

For JSON format, the envelope is embedded as a JSON object:

```json
{
  "bstr_version": 1,
  "model_type": "gbdt",
  "model": { ... }
}
```

### Model Type Discriminant

```rust
enum ModelTypeId {
    GBDT = 1,
    GBLinear = 2,
    DART = 3,      // Reserved for future DART support
    // Extensible via new variants
}
```

**Note**: DART (Dropouts meet Multiple Additive Regression Trees) schema is reserved but not yet defined. DART requires additional fields for dropout rate and normalization that affect inference semantics. Schema will be specified when DART training is implemented.

Unknown model type IDs result in `ReadError::UnsupportedModelType { type_id }`.

### Common Schema Types

```rust
enum TaskKindSchema {
    Regression,
    BinaryClassification,
    MulticlassClassification,
}

enum FeatureTypeSchema {
    Numerical,
    Categorical,
}
```

In JSON, these serialize as lowercase strings: `"regression"`, `"binary_classification"`, etc.

### GBDT Model Schema

**Precision note**: All floating-point values in the schema are `f64`.

```rust
struct GBDTModelSchema {
    meta: ModelMetaSchema,
    forest: ForestSchema,
    config: GBDTConfigSchema,
}

struct ModelMetaSchema {
    task: TaskKindSchema,
    num_features: usize,
    num_classes: Option<usize>,
    feature_names: Option<Vec<String>>,
    feature_types: Option<Vec<FeatureTypeSchema>>,
}

struct ForestSchema {
    n_groups: usize,
    base_score: Vec<f64>,  // Canonical location for base_score [n_groups]
    trees: Vec<TreeSchema>,
    tree_groups: Option<Vec<usize>>,
}
```

**Config requirement**: The persisted schema includes a required training config for lossless round-tripping. Converters that import external models must synthesize an equivalent config (or fail with a clear error if not representable).

### Tree Schema

**Sentinel value**: `u32::MAX` (0xFFFFFFFF) represents "no child" for `left_children` and `right_children` at leaf nodes.

```rust
struct TreeSchema {
    split_indices: Vec<u32>,
    split_thresholds: Vec<f32>,
    left_children: Vec<u32>,       // u32::MAX for leaf nodes
    right_children: Vec<u32>,      // u32::MAX for leaf nodes
    default_left: Vec<bool>,
    is_leaf: Vec<bool>,
    leaf_values: LeafValuesSchema,
    split_types: Vec<u8>,          // 0 = Numeric, 1 = Categorical
    categories: Option<CategoriesSchema>,
    linear_coefficients: Option<LinearCoefficientsSchema>,
    gains: Option<Vec<f32>>,       // Optional, useful for feature importance
    covers: Option<Vec<f32>>,      // Optional, useful for interpretability
}
```

**Gains and covers**: These are optional but recommended for interpretability. If present, they enable feature importance calculations and SHAP-style explanations. Models converted from XGBoost/LightGBM should preserve these when available.

```rust
enum LeafValuesSchema {
    Scalar(Vec<f32>),
    Vector { values: Vec<f32>, k: u32 },  // flattened [n_nodes * k]
}
```

**Vector leaves**: `k` is the output dimension per leaf. For multiclass, `k = n_classes`. For multi-target regression, `k = n_targets`. The semantic interpretation depends on `task`.

```rust
struct CategoriesSchema {
    node_offsets: Vec<u32>,
    category_data: Vec<u32>,  // bitset words (little-endian)
}

struct LinearCoefficientsSchema {
    // Sparse representation: each leaf has indices and coefficients
    node_offsets: Vec<u32>,      // offset into feature_indices/coefficients per node
    feature_indices: Vec<u32>,   // feature indices for non-zero coefficients
    coefficients: Vec<f32>,      // coefficient values
    intercepts: Vec<f32>,        // one per leaf node
}
```

**Linear coefficients convention**: The intercept is stored separately from coefficients. When predicting with linear leaves: `output = intercept + sum(coef[i] * feature[idx[i]])`.

### Validation Invariants

After deserialization, the following invariants are validated:

**Tree invariants**:

- All parallel arrays have the same length `n_nodes`: `split_indices`, `split_thresholds`, `left_children`, `right_children`, `default_left`, `is_leaf`, `split_types`
- For non-leaf nodes: `left_children[i] < n_nodes` and `right_children[i] < n_nodes`
- For leaf nodes: `left_children[i] == u32::MAX` and `right_children[i] == u32::MAX`
- `leaf_values` has exactly `n_nodes` entries (Scalar) or `n_nodes * k` entries (Vector)
- If `categories` is present: `node_offsets.len() == n_nodes`
- If `gains` is present: `gains.len() == n_nodes`
- If `covers` is present: `covers.len() == n_nodes`

**Forest invariants**:

- `tree_groups.len() == trees.len()`
- `base_score.len() == n_groups`
- All `tree_groups[i] < n_groups`

**Model invariants**:

- `meta.base_scores == forest.base_score`
- `meta.n_groups == forest.n_groups`

Validation failures result in `ReadError::Validation { message }` with a descriptive error.

### GBLinear Model Schema

```rust
struct GBLinearModelSchema {
    meta: ModelMetaSchema,
    config: Option<GBLinearConfigSchema>,
    weights: Vec<f32>,  // flattened [n_features + 1, n_groups]
    n_features: u32,
    n_groups: u32,
}
```

**GBLinear invariants**:

- `weights.len() == (n_features + 1) * n_groups`

### Subcomponent Serialization

To support inspection and visualization, individual components can be serialized independently:

```rust
use std::io::Cursor;

// Serialize just a forest (without model metadata) into a writer
let mut buf = Vec::new();
boosters::persist::write_json_into(&forest, &mut buf)?;

// Serialize a single tree into a writer
let mut buf = Vec::new();
boosters::persist::write_json_into(&tree, &mut buf)?;
```

This allows Python tools to:

- Parse the JSON and build tree plots
- Extract feature importance data
- Analyze tree structure

### API

#### Rust API

```rust
/// Unified streaming serialization/deserialization trait.
///
/// The core design avoids "save/load" helpers in the model types.
/// Instead models implement a single trait that can write/read via `Write`/`Read`.
/// The `Bstr` prefix is omitted since this is the only persistence format.
pub trait SerializableModel: Sized {
    /// Model type identifier stored in the header.
    const MODEL_TYPE: ModelTypeId;

    /// Write binary `.bstr` into any writer.
    fn write_into<W: std::io::Write>(&self, writer: W, opts: &WriteOptions) -> Result<(), WriteError>;

    /// Write JSON `.bstr.json` into any writer (UTF-8 JSON bytes).
    fn write_json_into<W: std::io::Write>(&self, writer: W, opts: &JsonWriteOptions) -> Result<(), WriteError>;

    /// Read binary `.bstr` from any reader.
    fn read_from<R: std::io::Read>(reader: R, opts: &ReadOptions) -> Result<Self, ReadError>;

    /// Read JSON `.bstr.json` from any reader.
    fn read_json_from<R: std::io::Read>(reader: R) -> Result<Self, ReadError>;
}

/// High-level helpers built on the trait (implemented once).
pub mod persist {
    pub fn write_into<M: SerializableModel, W: std::io::Write>(model: &M, writer: W, opts: &WriteOptions) -> Result<(), WriteError>;
    pub fn write_json_into<M: SerializableModel, W: std::io::Write>(model: &M, writer: W, opts: &JsonWriteOptions) -> Result<(), WriteError>;
    pub fn read_from<M: SerializableModel, R: std::io::Read>(reader: R, opts: &ReadOptions) -> Result<M, ReadError>;
    pub fn read_json_from<M: SerializableModel, R: std::io::Read>(reader: R) -> Result<M, ReadError>;
}

/// Polymorphic model reading (when model type is unknown)
pub enum Model {
    GBDT(GBDTModel),
    GBLinear(GBLinearModel),
}

impl Model {
    /// Read any model type from a reader, auto-detecting from the header.
    pub fn read_from<R: std::io::Read>(reader: R, opts: &ReadOptions) -> Result<Self, ReadError>;
}

/// Format type indicator from the header.
pub enum FormatType {
    /// JSON file (`.bstr.json`). This variant is detected by parsing JSON, not by a binary header byte.
    Json,
    /// Uncompressed MessagePack (format byte 0x01)
    MessagePack,
    /// Zstd-compressed MessagePack (format byte 0x02)
    MessagePackZstd,
}

/// Quick metadata inspection without full deserialization
pub struct ModelInfo {
    pub schema_version: u32,
    pub model_type: ModelTypeId,
    pub format: FormatType,
    pub payload_size: Option<u64>, // Payload bytes if known (e.g., seekable sources)
}

impl ModelInfo {
    /// Read only the header (32 bytes for binary, or parse JSON header).
    ///
    /// For seekable sources, implementations may also read the trailer to populate payload_size.
    pub fn inspect<R: std::io::Read>(reader: R) -> Result<Self, ReadError>;
}
```

#### Python API

```python
# Serialize/deserialize bytes (no file IO opinionated by the library)
b = model.to_bytes()             # binary bytes
j = model.to_json_bytes()        # UTF-8 JSON bytes

model = GBDTModel.from_bytes(b)
model = GBDTModel.from_json_bytes(j)

# Polymorphic decode from bytes
from boosters import loads, inspect
m = loads(b)  # Returns GBDTModel, GBLinearModel, etc.

# Quick inspection without full deserialization
info = inspect(b)
print(f"Model type: {info.model_type}, version: {info.schema_version}")

# Inspect model structure
tree = model.get_tree(0)
print(tree.to_dict())  # Python dict for visualization

# Convert from XGBoost/LightGBM (Python-only utility, JSON format only)
# Users who need binary format load the JSON and re-export via the model API.
from boosters.convert import xgboost_to_json_bytes, lightgbm_to_json_bytes

j = xgboost_to_json_bytes("xgb_model.json")     # From file path
j = xgboost_to_json_bytes(xgb_booster)          # From loaded Booster object

j = lightgbm_to_json_bytes("lgb_model.txt")
j = lightgbm_to_json_bytes(lgb_booster)

# To get binary format from an imported model:
model = GBDTModel.from_json_bytes(j)
b = model.to_bytes()  # Now you have compressed binary
```

### Python Schema Mirror (JSON)

To enable conversion tooling and explainability (e.g., plotting trees) without writing custom JSON deserializers, the Python package provides a mirror of the Rust schema types for the JSON format.

Key idea: users can do `json.loads(...)` and validate/parse into a typed model using datamodels.

**Scope**:

- The schema mirror targets **JSON** (`.bstr.json`) only.
- Binary `.bstr` parsing in Python is provided by the Rust extension module (PyO3). A pure-Python binary parser is out of scope.

**Proposed Python module**: `packages/boosters-python/src/boosters/persist/schema.py`

- Default implementation uses `pydantic` models (recommended) for easy parsing (`ModelFile.model_validate(...)`), validation (type/shape checks), and stable JSON round-tripping (`model_dump()` / `model_dump_json()`).
- If we want to keep `pydantic` optional, we can gate it behind an extra (e.g. `boosters[schema]`) and otherwise expose the raw dict form.

Example usage:

```python
import json

from boosters.persist.schema import ModelFile

j = open("model.bstr.json", "rb").read()
data = json.loads(j)
mf = ModelFile.model_validate(data)  # pydantic v2

tree0 = mf.model.forest.trees[0]
print(tree0.split_indices)
```

This is also the foundation for explainability helpers:

- `TreeSchema -> networkx` conversion
- matplotlib/plotly tree plotters
- feature importance extractors

**File I/O policy**:

- Rust uses `Read`/`Write` based APIs; callers decide whether to write to a file, buffer, socket, etc.
- Python returns/accepts `bytes`; callers decide where bytes are stored.
- The `.bstr` and `.bstr.json` extensions are conventions for humans.

**Python exceptions**: Read errors raise `boosters.ReadError`, a subclass of `ValueError`. IO errors raise `IOError`/`OSError`.

**sklearn note**: The `.bstr` format is boosters' native persistence format. joblib/pickle are not supported. Use `to_bytes()` / `from_bytes()` for model persistence.

**Python options**: In v1, Python uses defaults (compression level 3, compact JSON). Options are not exposed as kwargs. Power users can use Rust for custom settings.

### Error Handling

```rust
enum ReadError {
    /// File not found or IO error
    Io(std::io::Error),
    /// Invalid magic bytes
    InvalidMagic,
    /// Unsupported schema version (too new)
    UnsupportedVersion { found: u32, max_supported: u32 },
    /// Unknown model type in file
    UnsupportedModelType { type_id: u8 },
    /// Checksum mismatch (file corrupted)
    ChecksumMismatch { expected: u32, found: u32 },
    /// Decompression failed (invalid zstd data)
    Decompression(String),
    /// Deserialization failed (invalid MessagePack/JSON)
    Deserialize(String),
    /// Model validation failed (invariant violated)
    Validation(String),
}

enum WriteError {
    /// IO error
    Io(std::io::Error),
    /// Serialization failed
    Serialize(String),
}
```

### Migration: Compat Layer Removal

The compat layer in `crates/boosters/src/compat` will be removed:

**Files to delete**:

- `crates/boosters/src/compat/mod.rs`
- `crates/boosters/src/compat/xgboost/` (entire directory)
- `crates/boosters/src/compat/lightgbm/` (entire directory)

**Features to remove from Cargo.toml**:

- `xgboost-compat`
- `lightgbm-compat`

**Test migration**:

1. Convert existing XGBoost JSON test cases to `.bstr.json` format.
2. Update integration tests to load `.bstr` models directly.
3. Keep Python conversion utilities for users who need to import XGBoost/LightGBM models.

**Python conversion utilities** (new module: `packages/boosters-python/src/boosters/convert.py`):

- `xgboost_to_json_bytes(path_or_booster) -> bytes`
- `lightgbm_to_json_bytes(path_or_booster) -> bytes`

Optionally (for conversion + explainability tooling), expose schema-producing helpers:

- `xgboost_to_schema(path_or_booster) -> boosters.persist.schema.ModelFile`
- `lightgbm_to_schema(path_or_booster) -> boosters.persist.schema.ModelFile`

**Conversion principle**: Converters output JSON-only (the human-readable interchange format). They must not instantiate boosters runtime model types. Users who want binary format load the JSON into a model and re-export:

```python
from boosters import GBDTModel
from boosters.convert import xgboost_to_json_bytes

j = xgboost_to_json_bytes("xgb_model.json")
model = GBDTModel.from_json_bytes(j)
b = model.to_bytes()  # Binary compressed format
```

### Module Organization

```text
crates/boosters/src/persist/
├── mod.rs              # Public API: writer/reader based entrypoints
├── schema.rs           # Schema types (GBDTModelSchema, TreeSchema, etc.)
├── envelope.rs         # Binary envelope parsing/writing
├── json.rs             # JSON format implementation
├── binary.rs           # MessagePack + zstd implementation
├── error.rs            # ReadError, WriteError
├── migrate.rs          # Schema version migration functions
└── convert.rs          # Model <-> Schema conversions (From/TryFrom impls)
```

**Conversion traits**: Schema types are separate from runtime types. Conversion is encapsulated in `convert.rs`:

```rust
// In persist/convert.rs
impl From<&GBDTModel> for GBDTModelSchema { ... }
impl TryFrom<GBDTModelSchema> for GBDTModel { ... }

impl From<&Forest<ScalarLeaf>> for ForestSchema { ... }
impl TryFrom<ForestSchema> for Forest<ScalarLeaf> { ... }
```

### Public API Surface

The `persist` module is behind the `persist` crate feature (enabled by default). Public exports:

```rust
// boosters::persist — primary API
pub use persist::{
    write_into, write_json_into,
    read_from, read_json_from,
    ReadError, WriteError,
    ReadOptions, WriteOptions, JsonWriteOptions,
    Model, ModelInfo,
    SCHEMA_VERSION,
    SerializableModel,
};

// boosters::persist::schema — for advanced users
pub mod schema {
    pub use super::schema::{
        GBDTModelSchema, GBLinearModelSchema,
        ForestSchema, TreeSchema,
        ModelMetaSchema, TaskKindSchema, FeatureTypeSchema,
        // ... all schema types
    };
}
```

**Options types**:

```rust
/// Options for reading binary `.bstr` files.
pub struct ReadOptions {
    /// Skip checksum verification (not recommended, for benchmarking only).
    pub skip_checksum: bool,
}

/// Options for writing binary `.bstr` files.
pub struct WriteOptions {
    /// Compression level: 0 = uncompressed MessagePack (format byte 0x01),
    /// 1-22 = zstd compression levels (format byte 0x02). Default: 3.
    pub compression_level: u8,
}

/// Options for writing JSON `.bstr.json` files.
pub struct JsonWriteOptions {
    /// Pretty-print with indentation (default: false for compact output).
    pub pretty: bool,
}
```

All options types implement `Default` with sensible values.

Schema types are public to allow advanced use cases:

- Direct JSON manipulation for tooling
- Custom model builders for testing
- Integration with external visualization tools

### Files

| Path | Purpose |
| ---- | ------- |
| `crates/boosters/src/persist/mod.rs` | New module root |
| `crates/boosters/src/persist/schema.rs` | Schema types with Serde derives |
| `crates/boosters/src/persist/envelope.rs` | Envelope parsing and writing |
| `crates/boosters/src/persist/json.rs` | JSON format implementation |
| `crates/boosters/src/persist/binary.rs` | MessagePack + zstd implementation |
| `crates/boosters/src/persist/error.rs` | Error types |
| `crates/boosters/src/persist/migrate.rs` | Schema version migration |
| `crates/boosters/src/persist/convert.rs` | Model ↔ Schema conversions |
| `packages/boosters-python/src/boosters/convert.py` | XGBoost/LightGBM converters |

### Dependencies

New crate dependencies for the `persist` module:

| Crate | Version | Purpose | Optional |
| ----- | ------- | ------- | -------- |
| `serde_json` | ^1.0 | JSON serialization | No |
| `rmp-serde` | ^1.3 | MessagePack serialization | No |
| `crc32c` | ^0.6 | CRC32C checksum | No |
| `zstd` | ^0.13 | Compression | Yes (via `persist`) |

**Feature flags**:

- `persist` (default: enabled): Enables the native `.bstr` persistence module, including JSON, MessagePack, CRC32C, and zstd compression.

There is intentionally a single persistence feature gate: when `persist` is enabled, both reading and writing of compressed binary payloads (format byte `0x02`) are supported.

### Usage Examples

#### Rust

```rust
use std::fs::File;

use boosters::{
    GBDTModel,
    persist::{
        BinaryReadOptions, BinaryWriteOptions, JsonWriteOptions, Model, ModelInfo,
        SerializableModel,
    },
};

// Write a trained model
let model: GBDTModel = train_model(&data)?;
let mut out = File::create("model.bstr")?;
model.write_into(&mut out, &BinaryWriteOptions::default())?;

let mut out_json = File::create("model.bstr.json")?;
model.write_json_into(&mut out_json, &JsonWriteOptions::pretty())?;

// Read a model
let mut inp = File::open("model.bstr")?;
let loaded = GBDTModel::read_from(&mut inp, &BinaryReadOptions::default())?;

// Polymorphic read (when model type is unknown)
match Model::load("unknown.bstr")? {
    Model::GBDT(m) => println!("GBDT with {} trees", m.forest().n_trees()),
    Model::GBLinear(m) => println!("Linear model"),
}

// Quick inspection without full read (header-only)
let info = ModelInfo::inspect_file("model.bstr")?;
println!("Version: {}, Type: {:?}", info.schema_version, info.model_type);

// Serialize to bytes (for network transfer, caching)
let mut bytes = Vec::new();
model.write_into(&mut bytes, &BinaryWriteOptions::default())?;
let restored = GBDTModel::read_from(bytes.as_slice(), &BinaryReadOptions::default())?;
```

#### Python

```python
import boosters
from boosters import GBDTModel

# Bytes-based: users decide whether to write to disk, send over network, etc.
b = model.to_bytes()
open("model.bstr", "wb").write(b)

loaded = GBDTModel.from_bytes(open("model.bstr", "rb").read())

# Polymorphic parse (returns appropriate model type)
model = boosters.loads(open("model.bstr", "rb").read())

# Quick inspection
info = boosters.inspect(open("model.bstr", "rb").read())
print(f"Schema v{info.schema_version}, type={info.model_type}")

# Convert from XGBoost (JSON-only; load & re-export for binary)
from boosters.convert import xgboost_to_json_bytes
j = xgboost_to_json_bytes("xgboost_model.json")
model = GBDTModel.from_json_bytes(j)
open("converted.bstr", "wb").write(model.to_bytes())
```

### Integration

| Component | Integration Point |
| --------- | ----------------- |
| `GBDTModel` | Implements `SerializableModel` |
| `GBLinearModel` | Implements `SerializableModel` |
| `Model` | Polymorphic `read_from()` |
| `ModelInfo` | `inspect()` for header-only read |
| `Forest<L>` | `Serialize`, `Deserialize` derives |
| `Tree<L>` | `Serialize`, `Deserialize` derives |
| `LinearModel` | `Serialize`, `Deserialize` derives |
| Python bindings | `to_bytes()`, `from_bytes()`, `loads()`, `inspect()`, `convert` module |

## Testing

### Test Strategy

| Test Type | Coverage |
| --------- | -------- |
| Round-trip | Write → Read preserves all model data |
| Format detection | Auto-detect JSON vs binary |
| Version compatibility | Read old versions with new library |
| Checksum validation | Reject corrupted files |
| Error messages | Clear errors for unsupported versions |
| Subcomponent | Serialize/deserialize trees, forests independently |
| Python integration | Write in Rust, read in Python and vice versa |
| Polymorphic read | `Model::read_from()` returns correct type |
| Fuzz testing | Binary parser handles malformed input safely |
| Conversion pipeline | XGBoost JSON → boosters JSON → binary → read back |

### Comparison Strategy

Round-trip tests verify data preservation with per-field comparison:

| Field Type | Comparison Method |
| ---------- | ----------------- |
| `f32` values | Absolute difference ≤ 1e-7 (float serialization tolerance) |
| `f64` values | N/A (schema uses f32 for inference) |
| Integer arrays | Exact equality |
| Boolean arrays | Exact equality |
| Categorical bitsets | Exact equality (bit-for-bit) |
| Strings | Exact equality |
| Optional fields | Both None or both Some with value equality |

### Edge Cases

The following edge cases must have explicit test coverage:

- **Empty forest**: GBDT model with 0 trees
- **Single-node tree**: Tree with only a root leaf (no splits)
- **All-categorical tree**: Tree with only categorical splits
- **Max-depth tree**: Tree at maximum supported depth
- **Large forest**: 1000+ trees (stress test)
- **Multi-output**: Vector leaves with k > 1
- **Linear leaves**: Tree with `LeafCoefficients` present
- **Missing optionals**: Model without feature_names, feature_types
- **Unicode**: Feature names with non-ASCII characters

### Version Compatibility Matrix

Maintain test fixtures for each schema version:

```text
tests/test-cases/persist/
├── v1/
│   ├── gbdt_scalar.bstr
│   ├── gbdt_vector.bstr
│   ├── gblinear.bstr
│   └── expected_outputs.json
└── v2/  # When we increment schema version
    └── ...
```

Each schema version directory contains:

1. Binary `.bstr` files
2. JSON `.bstr.json` files (for human inspection)
3. Expected prediction outputs for verification

### Fixture Generation Process

Test fixtures are generated and maintained as follows:

1. **Initial generation**: Run `cargo run --example persist_fixtures` to create fixtures for the current schema version
2. **Schema changes**: When incrementing schema version:
   - Commit current fixtures (they become backward compatibility tests)
   - Update schema with new version number
   - Generate new fixtures in `vN+1/` directory
3. **Regeneration**: Never regenerate old version fixtures—they are immutable once committed
4. **CI verification**: CI runs read tests on all version directories to ensure backward compatibility


Example fixture generator (in `examples/persist_fixtures.rs`):

```rust
fn main() -> Result<()> {
    let model = create_test_gbdt_model();
    let mut out = std::fs::File::create("tests/test-cases/persist/v1/gbdt_scalar.bstr")?;
    model.write_into(&mut out, &boosters::persist::WriteOptions::default())?;

    let mut out_json = std::fs::File::create("tests/test-cases/persist/v1/gbdt_scalar.bstr.json")?;
    model.write_json_into(&mut out_json)?;
    // ... generate other fixtures
}
```

### Cross-Platform Testing

Binary format is little-endian. CI should verify:

- Files written on macOS (ARM64) load on Linux (x86_64)
- Files written on Linux load on macOS
- This is covered by having CI run on multiple platforms with shared test fixtures

### Corrupted File Testing

Test graceful handling of malformed input:

| Test Case | Expected Error |
| --------- | -------------- |
| Truncated file (< 32 bytes) | `Io` or `InvalidMagic` |
| Wrong magic bytes | `InvalidMagic` |
| Valid header, wrong trailer checksum | `ChecksumMismatch` |
| Valid header, truncated payload or missing trailer | `Deserialize` |
| Valid header, invalid zstd payload | `Decompression` |
| Valid structure, invalid tree (bad child index) | `Validation` |

## Changelog

- 2026-01-02: Marked as Accepted; linked to implementation backlog
- 2026-01-02: Updated schema spec to match implementation (config required; removed best_iteration/eval history; clarified f64 precision)

### Options Testing

Test behavior of various option values:

| Option | Value | Expected Behavior |
| ------ | ----- | ----------------- |
| `WriteOptions::compression_level` | 0 | Output is uncompressed MessagePack (format byte 0x01) |
| `WriteOptions::compression_level` | 3 | Output is zstd-compressed (format byte 0x02) |
| `WriteOptions::compression_level` | 23+ | `WriteError` (invalid compression level) |
| `ReadOptions::skip_checksum` | true | Read succeeds even with invalid trailer checksum |
| `ReadOptions::skip_checksum` | false | Read fails with `ReadError::ChecksumMismatch` on bad checksum |
| `JsonWriteOptions::pretty` | true | JSON output contains newlines and indentation |
| `JsonWriteOptions::pretty` | false | JSON output is compact (no extra whitespace) |

### Property-Based Testing

Use `proptest` or `quickcheck` for round-trip tests with randomly generated models:

```rust
proptest! {
    #[test]
    fn roundtrip_gbdt(model in arb_gbdt_model()) {
        let mut bytes = Vec::new();
        model.write_into(&mut bytes, &boosters::persist::WriteOptions::default()).unwrap();
        let loaded = GBDTModel::read_from(bytes.as_slice(), &boosters::persist::ReadOptions::default()).unwrap();
        assert_models_equal(&model, &loaded);
    }
}
```

Arbitrary model generators should produce:

- Variable tree depths (1-20)
- Variable forest sizes (1-100 trees)
- Mix of scalar and vector leaves
- Optional categorical splits and linear coefficients

### Validation Failure Testing

Test each validation invariant with a targeted invalid model:

- Tree with mismatched array lengths
- Tree with out-of-bounds child index
- Forest with mismatched tree_groups length
- Model with inconsistent base_scores

### Performance Benchmarks

Establish baseline performance targets:

| Operation | Model Size | Target Time | Notes |
| --------- | ---------- | ----------- | ----- |
| Write (binary+zstd) | 100 trees × 1K nodes | < 20ms | Level 3 compression |
| Write (binary+zstd) | 1000 trees × 1K nodes | < 200ms | Level 3 compression |
| Read (binary+zstd) | 100 trees × 1K nodes | < 10ms | Including decompression |
| Read (binary+zstd) | 1000 trees × 1K nodes | < 100ms | Including decompression |
| Write (JSON) | 100 trees × 1K nodes | < 50ms | Larger output |
| Inspect (header) | Any size | < 1ms | Only reads 32 bytes |

Benchmarks are documented in `benches/persist.rs` and run as part of CI.

### CI Requirements

The following CI checks are required for the `persist` module:

| Check | Description |
| ----- | ----------- |
| Backward compatibility | Read all versioned fixtures (v1, v2, ...) successfully |
| Cross-language | Write in Rust, read in Python; write in Python, read in Rust |
| Cross-platform | Test fixtures committed in CI read on all platforms |
| Coverage | `persist` module maintains ≥ 90% test coverage |
| Fuzz testing | Weekly fuzz runs on binary parser (OSS-Fuzz or cargo-fuzz) |
| Regression artifacts | Buggy files added as permanent test fixtures |

**Backward compatibility is a release blocker**: Removing support for loading an old schema version requires a major version bump.

**Fixture management**: All test fixtures are committed to the repository (not generated at test time). This ensures reproducibility and catches regressions when fixture generation code changes.

## Alternatives

### Alternative 1: Use Protocol Buffers

**Rejected**: Adds a build-time dependency (protoc), complicates the build for users, and doesn't provide significant benefits over MessagePack for our use case.

### Alternative 2: Use FlatBuffers

**Rejected**: Zero-copy access is not needed (we always deserialize fully), and the schema tooling adds complexity.

### Alternative 3: Keep Compat Layer

**Rejected**: The compat layer is maintenance overhead that doesn't benefit users training with boosters. Python utilities are sufficient for import.

### Alternative 4: Use JSON Only

**Rejected**: Binary format is important for production (smaller files, faster loading). JSON alone would hurt deployment scenarios.

## Design Decisions

**DD-1: Schema versioning with monotonic version number.** Simple and predictable. Migration functions handle old → new conversions.

**DD-2: MessagePack over Protobuf/FlatBuffers.** No code generation, self-describing, and serde ecosystem integration.

**DD-3: Optional zstd compression.** Tree forests can be large; zstd provides excellent compression with minimal CPU overhead.

**DD-4: Checksum in envelope.** Detect corruption early before deserialization fails with confusing errors.

**DD-5: Config is required in schema.** This avoids lossy deserialization paths and hardcoded defaults during load. Converters must provide a representable config; unsupported/custom objectives or metrics should fail fast.

**DD-6: Sequential deserialization (no parallelism).** MessagePack is fundamentally sequential, and typical model sizes (< 1M nodes) load in under 100ms. Parallel tree deserialization adds significant complexity for marginal benefit. Deferred as future optimization if profiling shows deserialization as a bottleneck.

**DD-7: 32-byte envelope with reserved space.** Provides room for future envelope fields (e.g., additional checksums, feature flags) without breaking format compatibility. Reserved bytes must be zero on write and ignored on read.

**DD-8: Streaming read with end-of-stream checksum verification.** Readers compute CRC32C incrementally while decoding. The checksum is validated after the trailer is read; on mismatch, the read fails and the partially-built model is dropped. This avoids buffering the full payload and keeps `Read`-only sources supported.

## Security Considerations

The `persist` module parses untrusted input (files from disk or network). Security measures:

1. **Input validation**: All array lengths and indices are validated before use
2. **Memory limits**: Deserializer should set reasonable size limits (e.g., max 1GB payload)
3. **Fuzz testing**: Binary parser is fuzz-tested before each release
4. **Checksum verification**: Corrupted files are rejected early
5. **No code execution**: Schema contains only data, never code (no pickle-style risks)

**Before v1.0 release**: Complete at least 1 week of continuous fuzzing with cargo-fuzz or OSS-Fuzz.

## Open Questions

1. ~~**Should we support streaming large models?**~~ No. Revisit if models exceed available RAM.
2. ~~**Should binary format be default?**~~ Yes, with `.bstr` extension. JSON is opt-in with `.bstr.json`.
3. ~~**How to handle custom objectives?**~~ Store objective name as string; custom objectives require matching code at load time.

All open questions have been resolved.

## Future Work

- **CLI tool**: A `bstr` command-line tool for inspecting, validating, and converting model files.
- **Schema extensibility**: New tree node types (e.g., neural network leaves) should use `#[serde(other)]` or forward-compatible enum encoding to allow old readers to gracefully skip unknown variants.
- **Parallel deserialization**: If profiling shows deserialization is a bottleneck for very large forests, consider parallel tree deserialization.
- **Python options**: Expose `WriteOptions` / `JsonWriteOptions` as optional keyword arguments if users request fine-grained control.

## Appendix: JSON Schema Example

Example of a minimal GBDT model in `.bstr.json` format:

```json
{
  "bstr_version": 1,
  "model_type": "gbdt",
  "model": {
    "meta": {
      "task": "regression",
            "num_features": 10,
      "feature_names": ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"],
            "feature_types": ["numeric", "numeric", "numeric", "numeric", "numeric",
                                                "numeric", "numeric", "numeric", "numeric", "categorical"]
    },
    "config": {
            "objective": {"type": "squared_loss"},
            "metric": null,
            "n_trees": 1,
            "learning_rate": 0.1,
            "growth_strategy": {"type": "depth_wise", "max_depth": 6},
            "max_onehot_cats": 4,
            "lambda": 1.0,
            "alpha": 0.0,
            "min_child_weight": 1.0,
            "min_gain": 0.0,
            "min_samples_leaf": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "colsample_bylevel": 1.0,
            "binning": {
                "max_bins": 256,
                "sparsity_threshold": 0.9,
                "enable_bundling": true,
                "max_categorical_cardinality": 0,
                "sample_cnt": 200000
            },
            "linear_leaves": null,
            "early_stopping_rounds": null,
            "cache_size": 8,
            "seed": 42,
            "verbosity": "silent",
            "extra": {}
    },
    "forest": {
            "n_groups": 1,
      "base_score": [0.5],
      "trees": [
        {
                    "num_nodes": 7,
                    "split_indices": [3, 1, 5],
                    "thresholds": [0.5, 0.3, 0.7],
                    "children_left": [1, 3, 5],
                    "children_right": [2, 4, 6],
          "default_left": [true, true, false, false, false, false, false],
                    "leaf_values": {"type": "scalar", "values": [0.0, 0.0, 0.0, 0.1, -0.05, 0.08, -0.03]},
                    "gains": [100.5, 50.2, 30.1],
                    "covers": [1000.0, 600.0, 400.0]
        }
      ],
            "tree_groups": null
    }
  }
}
```

**Note**: `4294967295` is `u32::MAX`, representing "no child" at leaf nodes.

---

*This RFC should be linked from the `persist` module documentation for implementers seeking format details.*
