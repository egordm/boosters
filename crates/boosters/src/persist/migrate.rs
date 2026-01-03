//! Schema migration infrastructure.
//!
//! This module provides scaffolding for migrating schemas between versions.
//! Currently, only version 1 exists, so migrations are identity operations.
//!
//! # Future Migrations
//!
//! When adding a new schema version (e.g., v2):
//!
//! 1. Add `v1::` and `v2::` modules with version-specific schema types
//! 2. Implement `migrate_v1_to_v2()` transformation
//! 3. Update `migrate_gbdt_schema()` to chain migrations
//!
//! # Design Principles
//!
//! - **Forward-only**: Migrations go from older to newer versions only
//! - **Version detection**: Version is stored in file header/schema
//! - **No data loss**: Migrations must preserve all semantics
//! - **Default values**: New fields use sensible defaults

// Allow dead code since these are scaffolding for future migrations
#![allow(dead_code)]

use super::error::ReadError;
use super::schema::{GBDTModelSchema, GBLinearModelSchema};

/// Current schema version.
pub const CURRENT_VERSION: u32 = 1;

/// Migrate a GBDT schema to the current version.
///
/// Currently a no-op since only v1 exists.
///
/// # Arguments
///
/// * `schema` - The schema to migrate (any supported version)
///
/// # Returns
///
/// The schema upgraded to current version, or an error if migration fails.
///
/// # Example
///
/// ```ignore
/// let old_schema: GBDTModelSchema = serde_json::from_str(json)?;
/// let current = migrate_gbdt_schema(1, old_schema)?;
/// let model = GBDTModel::try_from(current)?;
/// ```
pub fn migrate_gbdt_schema(
    schema_version: u32,
    schema: GBDTModelSchema,
) -> Result<GBDTModelSchema, ReadError> {
    match schema_version {
        1 => migrate_gbdt_v1_to_v1(schema),
        v => Err(ReadError::UnsupportedVersion {
            version: v,
            max_supported: CURRENT_VERSION,
        }),
    }
}

/// Migrate a GBLinear schema to the current version.
///
/// Currently a no-op since only v1 exists.
pub fn migrate_gblinear_schema(
    schema_version: u32,
    schema: GBLinearModelSchema,
) -> Result<GBLinearModelSchema, ReadError> {
    match schema_version {
        1 => migrate_gblinear_v1_to_v1(schema),
        v => Err(ReadError::UnsupportedVersion {
            version: v,
            max_supported: CURRENT_VERSION,
        }),
    }
}

// =============================================================================
// Version 1 → Version 1 (Identity)
// =============================================================================

/// Identity migration for v1 GBDT schemas.
///
/// This exists as scaffolding for future migrations.
/// When v2 is added, this becomes `migrate_v1_to_v2`.
fn migrate_gbdt_v1_to_v1(schema: GBDTModelSchema) -> Result<GBDTModelSchema, ReadError> {
    // Identity - no transformation needed
    Ok(schema)
}

/// Identity migration for v1 GBLinear schemas.
fn migrate_gblinear_v1_to_v1(
    schema: GBLinearModelSchema,
) -> Result<GBLinearModelSchema, ReadError> {
    // Identity - no transformation needed
    Ok(schema)
}

// =============================================================================
// Migration Pattern Documentation
// =============================================================================

/// # Adding a New Schema Version
///
/// When schema changes are needed, follow this pattern:
///
/// ## 1. Version the old schema types
///
/// ```ignore
/// // Move current types to v1 module
/// mod v1 {
///     pub struct GBDTModelSchemaV1 { ... }
/// }
///
/// // Add new types in v2 module  
/// mod v2 {
///     pub struct GBDTModelSchemaV2 { ... }
/// }
///
/// // Current = v2
/// pub type GBDTModelSchema = v2::GBDTModelSchemaV2;
/// ```
///
/// ## 2. Implement migration function
///
/// ```ignore
/// fn migrate_gbdt_v1_to_v2(old: v1::GBDTModelSchemaV1) -> v2::GBDTModelSchemaV2 {
///     v2::GBDTModelSchemaV2 {
///         version: 2,
///         // Copy existing fields
///         meta: old.meta,
///         forest: old.forest,
///         // New fields get defaults
///         new_field: Default::default(),
///     }
/// }
/// ```
///
/// ## 3. Chain migrations
///
/// ```ignore
/// pub fn migrate_gbdt_schema(schema_version: u32, schema: RawSchema) -> Result<GBDTModelSchema, ReadError> {
///     match schema_version {
///         1 => {
///             let v1 = parse_v1(schema)?;
///             let v2 = migrate_gbdt_v1_to_v2(v1);
///             Ok(v2)
///         }
///         2 => parse_v2(schema),
///         v => Err(ReadError::UnsupportedVersion { ... }),
///     }
/// }
/// ```
///
/// ## 4. Update tests
///
/// - Add regression test with v1 fixtures
/// - Verify v1 files still load correctly after v2 is added
/// - Test that migrated schemas produce correct predictions
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{GBDTModel, GBLinearModel, ModelMeta};
    use crate::repr::gbdt::Forest;
    use crate::repr::gblinear::LinearModel;
    use crate::scalar_tree;
    use ndarray::array;

    #[test]
    fn migrate_gbdt_v1_identity() {
        let tree = scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => leaf(2.0),
        };
        let mut forest = Forest::for_regression().with_base_score(vec![0.5]);
        forest.push_tree(tree, 0);
        let meta = ModelMeta::for_regression(2);
        let model = GBDTModel::from_forest(forest, meta);

        let schema = GBDTModelSchema::from(&model);
        let migrated = migrate_gbdt_schema(1, schema.clone()).unwrap();

        // Should be identical
        assert_eq!(migrated.forest.trees.len(), schema.forest.trees.len());
    }

    #[test]
    fn migrate_gblinear_v1_identity() {
        let weights = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
        let linear = LinearModel::new(weights);
        let meta = ModelMeta::for_binary_classification(2);
        let model = GBLinearModel::from_linear_model(linear, meta);

        let schema = GBLinearModelSchema::from(&model);
        migrate_gblinear_schema(1, schema.clone()).unwrap();
    }

    #[test]
    fn migrate_unsupported_version() {
        // Create a minimal v99 schema to test version rejection
        let tree = scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => leaf(2.0),
        };
        let mut forest = Forest::for_regression().with_base_score(vec![0.5]);
        forest.push_tree(tree, 0);
        let meta = ModelMeta::for_regression(2);
        let model = GBDTModel::from_forest(forest, meta);

        let schema = GBDTModelSchema::from(&model);
        let err = migrate_gbdt_schema(99, schema).unwrap_err();
        assert!(matches!(
            err,
            ReadError::UnsupportedVersion { version: 99, .. }
        ));
    }
}
