# Backlog: RFC-0016 Model Serialization

**RFC**: [docs/rfcs/0016-model-serialization.md](../rfcs/0016-model-serialization.md)  
**Created**: 2026-01-02  
**Status**: Refined (Ready for Implementation)

---

## Overview

Implement native `.bstr` model serialization format for boosters, including:

- Rust `persist` module with binary and JSON formats
- Python schema mirror for explainability tooling
- Python conversion utilities (XGBoost/LightGBM → `.bstr.json`)
- Removal of Rust compat layer
- Migration of test fixtures from XGBoost JSON to native `.bstr`

### Success Metrics

- All models save/load correctly with round-trip verification
- Binary format 50%+ smaller than pretty-printed JSON
- Load performance <10ms for typical models (100 trees, 1K nodes)
- No compat layer code remains in codebase
- Python schema can parse any valid `.bstr.json`

### Out of Scope

- Version migration (V1 → V2) - deferred until V2 is needed
- Model editing/mutation after loading
- Forward compatibility (V1 cannot read future versions)
- External format support beyond XGBoost and LightGBM

---

## Milestones

| Milestone | Epics | Description |
|-----------|-------|-------------|
| M1: Core Persist | Epic 1 | Rust persist module complete with JSON and binary formats |
| M2: Python Usable | Epics 2, 3, 4 | Python bindings, schema mirror, and conversion utilities |
| M3: Compat Removed | Epics 5, 6 | Compat layer deleted, fixtures migrated, deps updated |
| M4: Release Ready | Epic 7 | QA complete, performance validated, ready for release |

### Parallelization Opportunities

- **Epics 3-4** (Python Schema + Converters) can start once Story 1.3 (JSON format) is complete
- **Epic 2** (Python Bindings) requires Story 1.5 (SerializableModel trait)
- **Epic 5** and **Epic 6** can be partially parallelized but 5.4 (compat removal) must wait for 6.1-6.2

### Effort Size Reference

| Size | Duration | Examples |
|------|----------|----------|
| S | 1-2 hours | Add exports, update docs, simple tests |
| M | 2-4 hours | New type definitions, integration tests |
| L | 4-8 hours | Binary parser, trait implementation, complex migration |
| XL | 1-2 days | Converter implementation, fuzz testing setup |

---

## Epic 1: Core Persist Module (Rust)

Implement the `crates/boosters/src/persist/` module with schema types, binary/JSON encoding, and the `SerializableModel` trait.

### Story 1.1: Schema Types and Envelope [M]

Implement the schema types and binary envelope structures.

**Tasks**:

- [x] 1.1.1: Create `persist/mod.rs` with module structure and public exports
- [x] 1.1.2: Create `persist/schema.rs` with all schema types (GBDTModelSchema, TreeSchema, ForestSchema, etc.) with serde derives
- [x] 1.1.3: Create `persist/envelope.rs` with BstrHeader, BstrTrailer, and parsing/writing functions
- [x] 1.1.4: Create `persist/error.rs` with ReadError, WriteError enums
- [x] 1.1.5: Update `src/lib.rs` to add `pub mod persist` and re-export key types
- [x] 1.1.6: Unit tests for envelope parsing (valid header, invalid magic, future version)

**Definition of Done**:

- Schema types compile with serde derives
- Header/trailer can be written and read correctly
- All error variants defined with clear messages

### Story 1.2: Model ↔ Schema Conversions [L]

Implement conversions between runtime model types and schema types.

**Tasks**:

- [x] 1.2.1: Create `persist/convert.rs` with `From<&GBDTModel> for GBDTModelSchema`
- [x] 1.2.2: Implement `TryFrom<GBDTModelSchema> for GBDTModel` with validation
- [x] 1.2.3: Implement Forest and Tree conversions (scalar and vector leaves)
- [x] 1.2.4: Implement GBLinear model conversions
- [x] 1.2.5: Unit tests for round-trip conversions (model → schema → model)

**Definition of Done**:

- All model types have bidirectional conversion
- Validation errors are clear and actionable
- Round-trip preserves all fields exactly

### Story 1.3: JSON Format Implementation [M]

Implement JSON serialization for `.bstr.json` files.

**Tasks**:

- [x] 1.3.1: Create `persist/json.rs` with `write_json_into` and `read_json_from`
- [x] 1.3.2: Implement `JsonWriteOptions` (pretty print option)
- [x] 1.3.3: Implement `ModelInfo::inspect` for JSON format detection
- [x] 1.3.4: Unit tests for JSON round-trip (GBDT, GBLinear)
- [x] 1.3.5: Test pretty-print vs compact output

**Definition of Done**:

- JSON output matches RFC schema example
- JSON output is deterministic for stable fixtures (stable key ordering)
- Both compact and pretty-print modes work
- Round-trip preserves all data with float tolerance

### Story 1.4: Binary Format Implementation [XL]

Implement binary serialization with MessagePack and optional zstd compression.

**Tasks**:

- [x] 1.4.1: Add `rmp-serde`, `crc32c`, `zstd` dependencies to Cargo.toml
- [x] 1.4.2: Create `persist/binary.rs` with `write_into` and `read_from`
- [x] 1.4.3: Implement streaming CRC32C checksum computation
- [x] 1.4.4: Implement `WriteOptions` with compression level (0 = raw, 1-22 = zstd)
- [x] 1.4.5: Implement `ReadOptions` with skip_checksum option
- [x] 1.4.6: Implement trailer ring buffer for streaming decode
- [x] 1.4.7: Handle edge case: payload smaller than 16 bytes (valid for empty models)
- [x] 1.4.8: Unit tests for binary round-trip with compression levels 0, 3, and invalid (23+)
- [x] 1.4.9: Test checksum validation (corrupt payload should fail)
- [x] 1.4.10: Ensure `WriteOptions::default()` uses compression level 3 (RFC default)

**Definition of Done**:

- Binary format matches RFC specification exactly
- Compression/decompression works correctly
- Checksum catches corruption
- Invalid compression level returns WriteError

### Story 1.5: SerializableModel Trait and Integration [L]

Implement the unified `SerializableModel` trait and integrate with model types.

**Tasks**:

- [x] 1.5.1: Define `SerializableModel` trait in `persist/mod.rs`
- [x] 1.5.2: Implement trait for `GBDTModel`
- [x] 1.5.3: Implement trait for `GBLinearModel`
- [x] 1.5.4: Implement polymorphic `Model` enum with `read_binary` / `read_json` and `load` (format auto-detect)
- [x] 1.5.5: Implement `ModelInfo::inspect` for format auto-detect
- [x] 1.5.6: Integration tests: write GBDT, read as Model, verify type
- [x] 1.5.7: Integration tests: format mismatch (write JSON, read as binary; write binary, parse as JSON)
- [x] 1.5.8: Cross-language test setup: Rust writes, Python reads; Python writes, Rust reads

**Definition of Done**:

- All model types implement `SerializableModel`
- Polymorphic loading works correctly
- Format auto-detection works for binary/JSON
- `to_bytes()` minimizes allocations (pre-size buffer based on schema)

### Story 1.6: Feature Flags (Single `persist`) [S]

Align crate features with the persistence design: a single `persist` feature gate that enables all persistence dependencies (serde + JSON + MessagePack + CRC32C + zstd).

**Tasks**:

- [x] 1.6.1: Add `persist` feature flag and make persistence deps optional in Cargo.toml
- [x] 1.6.2: Gate `boosters::persist` module behind `#[cfg(feature = "persist")]`
- [x] 1.6.3: Ensure the crate builds with `--no-default-features` (persist disabled)
- [x] 1.6.4: Ensure the crate builds with `--features persist` (persist enabled)

**Definition of Done**:

- `persist` is the only persistence-related feature gate
- Crate builds cleanly with persist enabled and disabled

### Story 1.7: Review and Demo (Epic 1) [S]

**Tasks**:

- [x] 1.7.1: Stakeholder feedback check - review `workdir/tmp/stakeholder_feedback.md` for Epic 1 concerns
- [x] 1.7.2: Prepare demo showing: JSON round-trip, binary round-trip, compression size comparison (JSON vs msgpack vs zstd), polymorphic load
- [x] 1.7.3: Document demo in `workdir/tmp/development_review_<timestamp>_epic1.md`

**Definition of Done**:

- Demo executed with concrete metrics (file sizes, load times)
- Any feedback incorporated into subsequent stories

### Story 1.8: Migration Scaffolding [S]

Set up schema migration infrastructure for future versions.

**Tasks**:

- [x] 1.8.1: Create `persist/migrate.rs` with migration framework
- [x] 1.8.2: Implement identity migration for v1 → v1 (scaffolding)
- [x] 1.8.3: Document migration pattern for future version bumps

**Definition of Done**:

- Migration module exists and compiles
- Pattern is documented for adding future migrations

---

## Epic 2: Python Bindings

Expose serialization API in Python via PyO3 bindings.

### Story 2.1: Model Serialization Methods [M] ✅

Add serialization methods to Python model classes.

**Tasks**:

- [x] 2.1.1: Add `to_bytes()` method to `GBDTModel` returning binary bytes
- [x] 2.1.2: Add `to_json_bytes()` method returning UTF-8 JSON bytes
- [x] 2.1.3: Add `from_bytes(b: bytes)` classmethod to `GBDTModel`
- [x] 2.1.4: Add `from_json_bytes(b: bytes)` classmethod to `GBDTModel`
- [x] 2.1.5: Repeat for `GBLinearModel`
- [x] 2.1.6: Python tests for round-trip (train → serialize → deserialize → predict)

**Definition of Done**:

- All model types have bytes-based serialization
- Round-trip preserves predictions within tolerance (e.g. 1e-6)
- Methods have proper docstrings

### Story 2.2: Polymorphic Loading and Inspection [S] ✅

Add type-agnostic loading and inspection helpers.

**Tasks**:

- [x] 2.2.1: Add `boosters.Model.load_from_bytes(b: bytes)` returning appropriate model type
- [x] 2.2.2: Add `boosters.Model.inspect_bytes(b: bytes)` returning `ModelInfo` object
- [x] 2.2.3: Define `ModelInfo` Python class with schema_version, model_type, format fields
- [x] 2.2.4: Python tests for polymorphic loading

**Definition of Done**:

- `Model.load_from_bytes()` returns correct model type
- `Model.inspect_bytes()` works without full deserialization
- ModelInfo is properly exposed

### Story 2.3: Python ReadError Exception [S] ✅

Expose proper exception hierarchy.

**Tasks**:

- [x] 2.3.1: Create `boosters.ReadError` exception class (subclass of ValueError)
- [x] 2.3.2: Map Rust ReadError variants to Python exception with meaningful messages
- [x] 2.3.3: Test error messages for: invalid magic, checksum mismatch, unsupported version

**Definition of Done**:

- Errors are caught as `boosters.ReadError`
- Error messages are clear and actionable

### Story 2.4: Review and Demo (Epic 2) [S] ✅

**Tasks**:

- [x] 2.4.1: Stakeholder feedback check for Epic 2
- [x] 2.4.2: Review public API surface for naming consistency and ergonomics
- [x] 2.4.3: Demo showing: Python save/load cycle, polymorphic loading, error handling
- [x] 2.4.4: Document in `tmp/development_review_2026-01-03_epic2.md`

**Definition of Done**:

- Demo executed and documented
- Public API reviewed for consistency

---

## Epic 3: Python Schema Mirror

Provide Python dataclasses/pydantic models mirroring the Rust schema for native JSON parsing.

### Story 3.1: Schema Definitions [M]

Create pydantic v2 models matching Rust schema.

**Tasks**:

- [ ] 3.1.1: Create `packages/boosters-python/src/boosters/persist/__init__.py`
- [ ] 3.1.2: Create `packages/boosters-python/src/boosters/persist/schema.py`
- [ ] 3.1.3: Define `ModelFile`, `GBDTModelSchema`, `ForestSchema`, `TreeSchema`
- [ ] 3.1.4: Define `ModelMetaSchema`, `TaskKind`, `FeatureType` enums
- [ ] 3.1.5: Define `GBLinearModelSchema`, `LinearModelSchema`
- [ ] 3.1.6: Unit tests: parse JSON from Rust, validate with pydantic

**Definition of Done**:

- Python schema matches Rust schema exactly
- JSON from Rust parses successfully into Python models
- pydantic validation catches invalid data

### Story 3.2: Schema Round-Trip [S]

Ensure Python schema can serialize back to valid JSON.

**Tasks**:

- [ ] 3.2.1: Test `model_dump_json()` produces valid `.bstr.json`
- [ ] 3.2.2: Load Python-generated JSON in Rust, verify it parses
- [ ] 3.2.3: Cross-language round-trip test: Rust → JSON → Python → JSON → Rust
- [ ] 3.2.4: Use `tests/test-cases/persist/v1/gbdt_regression.bstr.json` as reference fixture

**Definition of Done**:

- Python-generated JSON is loadable in Rust
- No data loss in cross-language round-trip

### Story 3.3: Optional Pydantic Dependency [S]

Make pydantic optional via extras.

**Tasks**:

- [ ] 3.3.1: Add `[schema]` extra to pyproject.toml with pydantic dependency
- [ ] 3.3.2: Guard schema import with try/except, provide helpful error if missing
- [ ] 3.3.3: Test that boosters works without pydantic installed (schema module unavailable)

**Definition of Done**:

- Library installs and works without pydantic
- `pip install boosters[schema]` adds pydantic and enables schema module

### Story 3.4: Review and Demo (Epic 3) [S]

**Tasks**:

- [ ] 3.4.1: Stakeholder feedback check for Epic 3
- [ ] 3.4.2: Demo: parse JSON with pure Python, access tree structure, round-trip
- [ ] 3.4.3: Document in `workdir/tmp/development_review_<timestamp>_epic3.md`

**Definition of Done**:

- Demo executed and documented

---

## Epic 4: Python Conversion Utilities

Provide Python utilities for converting XGBoost/LightGBM models to `.bstr.json`.

### Story 4.1: XGBoost Converter [L]

Implement XGBoost → `.bstr.json` conversion.

**Tasks**:

- [ ] 4.1.1: Create `packages/boosters-python/src/boosters/convert.py`
- [ ] 4.1.2: Implement `xgboost_to_json_bytes(path_or_booster) -> bytes`
- [ ] 4.1.3: Implement `xgboost_to_schema(path_or_booster) -> ModelFile` (optional)
- [ ] 4.1.4: Handle file path input (read JSON) and Booster object input
- [ ] 4.1.5: Test with gbtree, gblinear, and dart model types
- [ ] 4.1.6: Test with multi-class and multi-output models
- [ ] 4.1.7: Compare predictions: original XGBoost vs boosters-loaded (tolerance 1e-6)

**Definition of Done**:

- XGBoost models convert to valid `.bstr.json`
- Converted models load correctly in boosters
- All XGBoost test cases from datagen covered
- Converter outputs JSON-only and does not require instantiating boosters runtime model types

### Story 4.2: LightGBM Converter [L]

Implement LightGBM → `.bstr.json` conversion.

**Tasks**:

- [ ] 4.2.1: Implement `lightgbm_to_json_bytes(path_or_booster) -> bytes`
- [ ] 4.2.2: Implement `lightgbm_to_schema(path_or_booster) -> ModelFile` (optional)
- [ ] 4.2.3: Handle text model file input and Booster object input
- [ ] 4.2.4: Test with regression, binary, and multiclass models
- [ ] 4.2.5: Test with linear tree models
- [ ] 4.2.6: Compare predictions: original LightGBM vs boosters-loaded (tolerance 1e-6)

**Definition of Done**:

- LightGBM models convert to valid `.bstr.json`
- Converted models load correctly in boosters
- All LightGBM test cases from datagen covered
- Converter outputs JSON-only and does not require instantiating boosters runtime model types

### Story 4.3: Review and Demo (Epic 4) [S]

**Tasks**:

- [ ] 4.3.1: Stakeholder feedback check for Epic 4
- [ ] 4.3.2: Demo: convert XGBoost model, load in boosters, compare predictions
- [ ] 4.3.3: Document in `workdir/tmp/development_review_<timestamp>_epic4.md`

**Definition of Done**:

- Demo executed and documented

---

## Epic 5: Test Fixture Migration

Migrate test cases from XGBoost JSON to native `.bstr` format and remove Rust compat layer.

### Story 5.1: Generate Native Fixtures [M]

Create `.bstr` and `.bstr.json` fixtures from existing XGBoost test cases.

**Tasks**:

- [ ] 5.1.1: Create `examples/persist_fixtures.rs` fixture generator
- [ ] 5.1.2: Generate GBDT scalar leaf fixtures (regression, binary, multiclass)
- [ ] 5.1.3: Generate GBDT vector leaf fixtures (multi-output)
- [ ] 5.1.4: Generate GBLinear fixtures
- [ ] 5.1.5: Generate edge case fixtures (empty forest, single-node tree, deep tree)
- [ ] 5.1.6: Create `tests/test-cases/persist/v1/` directory structure
- [ ] 5.1.7: Include expected prediction outputs for each fixture (for verification)
- [ ] 5.1.8: Add regression test that fails if committed fixtures change unexpectedly
- [ ] 5.1.9: Commit fixtures as immutable test data

**Definition of Done**:

- All fixture types generated
- Fixtures load correctly with persist module
- JSON fixtures human-readable and match RFC schema

### Story 5.2: Update Integration Tests [M]

Migrate integration tests from compat layer to native fixtures.

**Tasks**:

- [ ] 5.2.1: Update `tests/training.rs` to save/load with persist module
- [ ] 5.2.2: Update `tests/compat.rs` → `tests/persist.rs` with native fixtures
- [ ] 5.2.3: Update `tests/quality_smoke.rs` if it uses compat layer
- [ ] 5.2.4: Verify all tests pass with new fixtures

**Definition of Done**:

- All integration tests pass with native fixtures
- No tests depend on compat layer for fixture loading

### Story 5.3: Update Testing Utilities [S]

Update `src/testing/` module to use native format.

**Tasks**:

- [ ] 5.3.1: Update `src/testing/cases.rs` to load `.bstr` files instead of XGBoost JSON
- [ ] 5.3.2: Remove `#[cfg(feature = "xgboost-compat")]` guards from testing module
- [ ] 5.3.3: Update test helper functions

**Definition of Done**:

- Testing utilities work without compat features
- Test cases load from native format

### Story 5.4: Remove Rust Compat Layer [L]

Delete the compat layer and associated code.

**Tasks**:

- [ ] 5.4.1: Delete `crates/boosters/src/compat/` directory
- [ ] 5.4.2: Remove `xgboost-compat` and `lightgbm-compat` features from Cargo.toml
- [ ] 5.4.3: Update `src/lib.rs` to remove compat module exports
- [ ] 5.4.4: Remove compat-related dependencies if no longer needed
- [ ] 5.4.5: Update `tests/compat/` directory (delete or migrate)
- [ ] 5.4.6: Update CI configuration to remove compat features from default test runs
- [ ] 5.4.7: Update `src/lib.rs` doc comments that reference compat layer
- [ ] 5.4.8: Verify library compiles and all tests pass
- [ ] 5.4.9: Run full test suite with `--all-features` to catch hidden dependencies

**Definition of Done**:

- No compat layer code remains
- No references to `compat::` in codebase (except docs/changelog)
- Library compiles without compat features
- Default features updated as needed (e.g., keep `persist` enabled by default)
- All tests pass
- All CI checks green (lints, tests, clippy, formatting)

### Story 5.5: Update Benchmarks [S]

Update comparison benchmarks that use compat layer.

**Tasks**:

- [ ] 5.5.1: Update `benches/common/models.rs` to load native fixtures
- [ ] 5.5.2: Update comparison benchmarks to use Python for model conversion
- [ ] 5.5.3: Verify benchmarks run correctly

**Definition of Done**:

- Benchmarks run with native fixtures
- Comparison benchmarks still functional

### Story 5.6: Review and Demo (Epic 5) [S]

**Tasks**:

- [ ] 5.6.1: Stakeholder feedback check for Epic 5
- [ ] 5.6.2: Demo: removed compat layer, show reduced code size, all tests passing
- [ ] 5.6.3: Document in `workdir/tmp/development_review_<timestamp>_epic5.md`

**Definition of Done**:

- Demo executed and documented
- Code size reduction measured

---

## Epic 6: Update Dependent Packages

Update boosters-datagen and other packages affected by compat removal.

### Story 6.1: Update boosters-datagen [M]

Modify datagen to generate native `.bstr` fixtures.

**Tasks**:

- [ ] 6.1.1: Add Python `boosters` (boosters-python) dependency to boosters-datagen (uv workspace), ensure extension import works
- [ ] 6.1.2: Update `xgboost.py` to also output `.bstr.json` via conversion utility
- [ ] 6.1.3: Update `lightgbm.py` to also output `.bstr.json` via conversion utility
- [ ] 6.1.4: Add new `bstr` CLI command to generate native fixtures
- [ ] 6.1.5: Update README.md with new commands
- [ ] 6.1.6: Test fixture generation end-to-end

**Definition of Done**:

- datagen can produce native `.bstr` fixtures
- Existing XGBoost/LightGBM generation still works (for comparison)

### Story 6.2: Update boosters-eval [S]

Check if eval package needs updates.

**Tasks**:

- [ ] 6.2.1: Audit boosters-eval for compat layer usage
- [ ] 6.2.2: Update model loading to use native format or Python conversion
- [ ] 6.2.3: Verify eval benchmarks work correctly

**Definition of Done**:

- boosters-eval works with native format
- No dependency on removed compat features

### Story 6.3: Documentation Updates [S]

Update documentation to reflect new serialization API.

**Tasks**:

- [ ] 6.3.1: Update README.md with serialization examples
- [ ] 6.3.2: Update docs/README.md if needed
- [ ] 6.3.3: Add migration guide for users of compat layer
- [ ] 6.3.4: Plan version bump (0.x → next minor) for breaking change

**Definition of Done**:

- Documentation reflects new API
- Migration path is clear for existing users

### Story 6.4: Review and Demo (Epic 6) [S]

**Tasks**:

- [ ] 6.4.1: Stakeholder feedback check for Epic 6
- [ ] 6.4.2: Demo: full workflow from training to serialization to loading
- [ ] 6.4.3: Document in `workdir/tmp/development_review_<timestamp>_epic6.md`

**Definition of Done**:

- Demo executed and documented

---

## Epic 7: Quality Assurance and Release

Final testing, performance validation, and release preparation.

### Story 7.1: Property-Based Testing [M]

Add comprehensive property-based tests.

**Tasks**:

- [ ] 7.1.1: Add proptest dependency
- [ ] 7.1.2: Implement `arb_gbdt_model()` arbitrary model generator
- [ ] 7.1.3: Implement round-trip property test for GBDT
- [ ] 7.1.4: Implement round-trip property test for GBLinear
- [ ] 7.1.5: Run property tests with sufficient iterations
- [ ] 7.1.6: Add negative tests for validation (invalid node counts, malformed schemas)

**Definition of Done**:

- Property tests pass with 1000+ iterations
- No edge cases discovered

### Story 7.2: Fuzz Testing [L]

Set up fuzz testing for binary parser.

**Tasks**:

- [ ] 7.2.1: Create fuzz target for binary reader
- [ ] 7.2.2: Run cargo-fuzz for minimum 1 hour
- [ ] 7.2.3: Fix any crashes discovered
- [ ] 7.2.4: Add any crash inputs as regression fixtures

**Definition of Done**:

- No crashes in 1 hour of fuzzing
- Parser handles malformed input gracefully

### Story 7.3: Performance Benchmarks [M]

Validate performance targets from RFC.

**Tasks**:

- [ ] 7.3.1: Create `benches/persist.rs` benchmark suite
- [ ] 7.3.2: Benchmark write (100 trees, 1K nodes) - target <20ms
- [ ] 7.3.3: Benchmark write (1000 trees, 1K nodes) - target <200ms
- [ ] 7.3.4: Benchmark read (100 trees) - target <10ms
- [ ] 7.3.5: Benchmark inspect (header only) - target <1ms
- [ ] 7.3.6: Compare against compat layer load performance (baseline)
- [ ] 7.3.7: Document benchmark results in `docs/benchmarks/`

**Note**: Task 7.3.6 MUST run before Story 5.4 (compat removal) to capture baseline.

**Definition of Done**:

- All performance targets met
- Benchmark results documented

### Story 7.4: Cross-Platform Testing [S]

Verify binary format works across platforms.

**Tasks**:

- [ ] 7.4.1: Generate fixtures on macOS ARM64
- [ ] 7.4.2: Verify fixtures load on Linux x86_64 (CI)
- [ ] 7.4.3: Verify fixtures load on Windows x86_64 (CI if available)

**Definition of Done**:

- Same fixtures load on all platforms
- CI validates cross-platform compatibility

### Story 7.5: Final Retrospective [S]

**Tasks**:

- [ ] 7.5.1: Run retrospective for RFC-0016 implementation
- [ ] 7.5.2: Document outcomes in `workdir/tmp/retrospective.md`
- [ ] 7.5.3: Create follow-up backlog items for any improvements identified
- [ ] 7.5.4: Verify all RFC-0016 acceptance criteria are met
- [ ] 7.5.5: Update RFC-0016 status to "Implemented"

**Definition of Done**:

- Retrospective completed
- Lessons learned documented
- RFC marked as implemented

---

## Dependencies

| Story | Depends On |
|-------|------------|
| 1.2 | 1.1 |
| 1.3 | 1.1, 1.2 |
| 1.4 | 1.1, 1.2 |
| 1.5 | 1.3, 1.4 |
| 1.6 | 1.4 |
| 1.8 | 1.2 |
| 2.1 | 1.5 |
| 2.2 | 2.1 |
| 2.3 | 2.1 |
| 3.1 | 1.3 (JSON format finalized) |
| 3.2 | 3.1, 2.1 |
| 4.1 | 3.1 |
| 4.2 | 3.1 |
| 5.1 | 1.5 |
| 5.2 | 5.1 |
| 5.3 | 5.1 |
| 5.4 | 5.2, 5.3, 6.1, 6.2 |
| 5.5 | 5.4 |
| 6.1 | 4.1, 4.2 |
| 6.2 | 5.1 |
| 7.1 | 1.5 |
| 7.2 | 1.5 |
| 7.3 | 1.5 |
| 7.4 | 5.1 |
| 7.5 | All stories complete |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| MessagePack schema compatibility issues | Medium | High | Extensive round-trip testing, property tests |
| Performance regression from serde overhead | Low | Medium | Benchmark early, optimize if needed |
| Python pydantic version conflicts | Medium | Low | Support pydantic v2 only, gate behind extra |
| Breaking changes in dependent packages | Low | Medium | Update datagen and eval incrementally |
| Complex migration for existing test suites | Medium | Medium | Create batch migration script for XGBoost JSON → .bstr |

**Rollback plan**: Story 5.4 (compat removal) should be the final task before M3 milestone. If issues arise post-removal, revert the deletion commit and address issues before re-attempting.

---

## Changelog

- 2026-01-02: Initial backlog created from RFC-0016
- 2026-01-02: Round 1 - Added milestones, fixed Epic 5-6 ordering, expanded test coverage, added CI task
- 2026-01-02: Round 2 - Added effort sizes, edge case handling, lib.rs exports, stricter DoD
- 2026-01-02: Round 3 - Added all effort sizes, Story 1.8 dependency, benchmark comparison, rollback plan
- 2026-01-02: Round 4 - Added parallelization section, negative tests, RFC acceptance verification, version bump task
- 2026-01-02: Round 5 - Added success metrics, out of scope, compression level, regression tests, CI gates, API review
- 2026-01-02: Round 6 - Final polish: fixture specification, allocation note, status updated to Refined
