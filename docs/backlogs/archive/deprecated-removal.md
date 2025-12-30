# Deprecated Folder Removal Backlog

**Status**: In Progress  
**Created**: 2025-12-29  
**Goal**: Remove the deprecated/ folder entirely while maintaining all functionality

## Context

The deprecated folder contains ~7794 lines of code that was marked for removal after RFC-0018 implementation.

### Audit Findings

**deprecated/binned/** (~5500 lines): Old binned implementation
- `BinStorage`, `BinType`, `BundlingFeatures`, `GroupStrategy`, `BinningStrategy`
- **COMPLETELY UNUSED** outside deprecated/ - can be deleted immediately

**deprecated/*.rs** (~2300 lines): Core data types still in use
- `Dataset`, `DatasetBuilder` - Used by Python bindings and examples
- `FeaturesView`, `TargetsView`, `WeightsView` - Used by training interface
- `DataAccessor`, `SampleAccessor` - Used for data access patterns
- `DatasetSchema`, `FeatureType`, `FeatureMeta` - Used for metadata
- `Column`, `SparseColumn` - Data storage types

### Performance Issue (CRITICAL) - FIXED ✅

Previously: Covertype benchmark showed quality regression (~3.88 mlogloss vs 0.43 LightGBM)

**Root Cause**: Equal-width binning instead of quantile binning in `bin_numeric()`.

**Fix Applied**: Changed to quantile binning with midpoint boundaries (commit 24e9ba9).

**Current Results**:
- **boosters gbdt**: 0.4137±0.0092 mlogloss ✅ (BETTER THAN LightGBM)
- **lightgbm gbdt**: 0.4285±0.0074 mlogloss

## Epic 1: Critical Bug Fix - COMPLETE ✅

### Story 1.1: Investigate Covertype Bundling Issue ✅ DONE

**Status**: Complete  
**Resolution**: Root cause was binning, not bundling. Fixed by implementing quantile binning.

---

## Epic 2: Delete Unused Code

### Story 2.1: Delete deprecated/binned/

**Status**: Not Started  
**Estimate**: 15 min

**Description**: Delete the completely unused deprecated/binned/ folder.

**Tasks**:
1. Delete deprecated/binned/ folder
2. Remove re-exports in data/binned/mod.rs
3. Remove dead code in training/gbdt/histograms/mod.rs (DeprecatedFeatureView, convert_*)
4. Verify compilation

---

### Story 2.2: Move Core Types Out of deprecated/

**Status**: Not Started  
**Estimate**: 1 hour

**Description**: Move essential types to proper locations and delete deprecated/.

**Tasks**:
1. Move files to data/ root
2. Update all imports
3. Delete deprecated/ folder

---

## Epic 3: Validation - COMPLETE ✅

### Story 3.1: Python Bindings Verification ✅ DONE

**Status**: Complete  
**Result**: Python bindings compile and work correctly.

---

### Story 3.2: Full Benchmark Suite ✅ DONE

**Status**: Complete  
**Result**: All benchmarks pass. Covertype mlogloss = 0.41 (beats LightGBM's 0.43).
