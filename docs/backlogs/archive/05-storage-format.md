# Native Storage Format Backlog

**Source**: [RFC-0021: Native Storage Format](../rfcs/0021-storage-format.md)  
**Created**: 2025-12-19  
**Status**: ✅ COMPLETE (2025-06-18)

---

## Overview

This backlog implements native `.bstr` serialization for booste-rs models, enabling save/load functionality for all model types.

**Dependencies**: None (can start immediately)  
**Enables**: Model API (Epic 2), Python Bindings (Epic 3)

**Actual Effort**: ~4 hours

---

## Epic 1: Native Storage Format ✅

**Goal**: Implement binary serialization format for booste-rs models.

**Status**: COMPLETE

---

### Story 1.1: Implement Format Header ✅

**RFC Section**: RFC-0021 "Format Header"  
**Effort**: S (30min)
**Actual**: ~30min

**Description**: Implement the 32-byte binary header with magic number, version, model type, flags, and checksum.

**Tasks**:

- [x] 1.1.1 Created `src/io/mod.rs` and `src/io/native.rs`
- [x] 1.1.2 Created `FormatHeader` struct (32 bytes, packed)
- [x] 1.1.3 Implemented `MAGIC = b"BSTR"`
- [x] 1.1.4 Defined `ModelType` enum (Gbdt, Dart, GbLinear)
- [x] 1.1.5 Defined `FormatFlags` bitfield (COMPRESSED, HAS_CATEGORICAL, HAS_LINEAR_LEAVES)
- [x] 1.1.6 Implemented `FormatHeader::to_bytes() -> [u8; 32]`
- [x] 1.1.7 Implemented `FormatHeader::from_bytes() -> Result<Self, DeserializeError>`
- [x] 1.1.8 Implemented CRC32 checksum with crc32fast

**Files Created**:
- [src/io/mod.rs](../../src/io/mod.rs)
- [src/io/native.rs](../../src/io/native.rs)

**Tests**: 8 tests in native.rs

---

### Story 1.2: Implement Model Payloads ✅

**RFC Section**: RFC-0021 "GBDT Payload", "GBLinear Payload"  
**Effort**: M (1-2h)
**Actual**: ~1h

**Description**: Define Postcard-serializable payload structs for each model type.

**Tasks**:

- [x] 1.2.1 Added `postcard` (with alloc feature) and `crc32fast` dependencies
- [x] 1.2.2 Created `Payload` version-tagged enum: `enum Payload { V1(PayloadV1) }`
- [x] 1.2.3 Implemented `PayloadV1` with metadata and model variants
- [x] 1.2.4 Created `ModelMetadata` struct
- [x] 1.2.5 Created `GbdtPayload`, `ForestPayload`, `TreePayload`
- [x] 1.2.6 Created `GbLinearPayload`
- [x] 1.2.7 Created `CategoriesPayload` and `LinearLeavesPayload`
- [x] 1.2.8 All structs have `#[derive(Serialize, Deserialize)]`

**Files Created**:
- [src/io/payload.rs](../../src/io/payload.rs)

**Tests**: 3 tests in payload.rs

---

### Story 1.3: Implement Codec API ✅

**RFC Section**: RFC-0021 "Serialization Codec"  
**Effort**: M (2-3h)
**Actual**: ~1.5h

**Description**: Implement the read/write API with optional zstd compression.

**Tasks**:

- [x] 1.3.1 Added `zstd` dependency (behind `storage-compression` feature)
- [x] 1.3.2 Implemented `NativeCodec` struct with compression options
- [x] 1.3.3 Implemented `Payload::from_forest(forest) -> Payload` conversion
- [x] 1.3.4 Implemented `Payload::from_linear_model(model) -> Payload` conversion
- [x] 1.3.5 Implemented `serialize()` with optional compression
- [x] 1.3.6 Implemented `deserialize()` with checksum verification
- [x] 1.3.7 Implemented `Payload::into_forest()` and `Payload::into_linear_model()`
- [x] 1.3.8 Implemented streaming: `write_to(writer)`, `read_from(reader)`
- [x] 1.3.9 Defined `SerializeError` and `DeserializeError` types

**Files Created**:
- [src/io/convert.rs](../../src/io/convert.rs)

**Tests**: 5 tests in convert.rs

---

### Story 1.4: Integrate with Forest/LinearModel ✅

**Effort**: M (30min-1h)
**Actual**: ~30min

**Description**: Add `save()`/`load()` methods to model types.

**Tasks**:

- [x] 1.4.1 Implemented `Forest::save(path)` and `Forest::load(path)`
- [x] 1.4.2 Implemented `Forest::to_bytes()` and `Forest::from_bytes()`
- [x] 1.4.3 Same methods for `LinearModel`
- [x] 1.4.4 Added rustdoc with examples

**Tests**: 5 additional tests in convert.rs (10 total)

---

### Story 1.5: Create Format Version Corpus ✅

**Effort**: S (30min)
**Actual**: ~30min

**Description**: Create test fixtures for format compatibility testing.

**Tasks**:

- [x] 1.5.1 Created `tests/test-cases/native/` directory
- [x] 1.5.2 Created integration test file: `tests/native_format.rs`
- [x] 1.5.3 Generated fixtures: simple_forest, multi_tree_forest, categorical_forest, multiclass_forest, simple_linear, multioutput_linear
- [x] 1.5.4 Created `generate_fixtures` test to regenerate corpus
- [x] 1.5.5 Created `load_fixtures` test for backward compatibility

**Files Created**:
- [tests/native_format.rs](../../tests/native_format.rs)
- [tests/test-cases/native/README.md](../../tests/test-cases/native/README.md)
- 6 `.bstr` fixture files

**Tests**: 15 integration tests (14 run, 1 ignored generator)

---

## Summary

### Implementation Statistics

| Story | Tests | Lines Added |
|-------|-------|-------------|
| 1.1 Format Header | 8 | ~250 |
| 1.2 Model Payloads | 3 | ~180 |
| 1.3 Codec API | 5 | ~350 |
| 1.4 Integration | 5 | ~150 |
| 1.5 Format Corpus | 15 | ~300 |
| **Total** | **36** | **~1,230** |

### Features Added

- `storage` feature flag (enables postcard + crc32fast)
- `storage-compression` feature flag (adds zstd)

### Verification Checklist ✅

- [x] `cargo test --features storage` — all tests pass (36 new tests)
- [x] Round-trip tests for Forest (simple, multi-tree, categorical, multiclass)
- [x] Round-trip tests for LinearModel (simple, multioutput)
- [x] Error handling tests (type mismatch, corruption, truncation, empty data)
- [x] Fixture files generated and loadable
- [x] Rustdoc added to public APIs

---

**Next Epic**: [Model API and Python Bindings](06-model-api-and-python.md)

---

**Completed**: 2025-06-18
**Reviewed By**: PO, Architect, Senior Engineer, QA Engineer
