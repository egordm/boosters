//! Feature type definitions.
//!
//! This module defines the schema types that describe dataset structure.

use std::collections::HashMap;

/// Logical feature types.
///
/// Features are stored as `f32` regardless of type. The `FeatureType` indicates
/// how to interpret the values during binning and splitting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FeatureType {
    /// Continuous numeric feature.
    ///
    /// Missing values: `f32::NAN`
    #[default]
    Numeric,

    /// Categorical feature stored as float, interpreted as integer category ID.
    ///
    /// Missing values: `f32::NAN` (or negative values)
    /// Valid categories: `0.0, 1.0, 2.0, ..., n_categories-1.0`
    ///
    /// Values are cast to `i32` during binning: `category_id = value as i32`
    ///
    /// This matches XGBoost and LightGBM internal representation.
    Categorical,
}

impl FeatureType {
    /// Returns true if this is a categorical feature.
    #[inline]
    pub fn is_categorical(&self) -> bool {
        matches!(self, FeatureType::Categorical)
    }

    /// Returns true if this is a numeric feature.
    #[inline]
    pub fn is_numeric(&self) -> bool {
        matches!(self, FeatureType::Numeric)
    }
}

/// Metadata for a single feature.
#[derive(Clone, Debug, Default)]
pub struct FeatureMeta {
    /// Feature name (optional).
    pub name: Option<String>,

    /// Feature type.
    pub feature_type: FeatureType,
}

impl FeatureMeta {
    /// Create metadata for a numeric feature.
    pub fn numeric() -> Self {
        Self {
            name: None,
            feature_type: FeatureType::Numeric,
        }
    }

    /// Create metadata for a numeric feature with a name.
    pub fn numeric_named(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            feature_type: FeatureType::Numeric,
        }
    }

    /// Create metadata for a categorical feature.
    pub fn categorical() -> Self {
        Self {
            name: None,
            feature_type: FeatureType::Categorical,
        }
    }

    /// Create metadata for a categorical feature with a name.
    pub fn categorical_named(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            feature_type: FeatureType::Categorical,
        }
    }

    /// Set the feature name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Schema describing the dataset structure.
///
/// Contains per-feature metadata and optional name-to-index mapping.
#[derive(Clone, Debug, Default)]
pub struct DatasetSchema {
    /// Per-feature metadata.
    features: Vec<FeatureMeta>,

    /// Feature name → index mapping (built lazily on first lookup).
    name_index: Option<HashMap<String, usize>>,
}

impl DatasetSchema {
    /// Create an empty schema.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a schema with the given feature metadata.
    pub fn from_features(features: Vec<FeatureMeta>) -> Self {
        Self {
            features,
            name_index: None,
        }
    }

    /// Create a schema where all features are numeric.
    pub fn all_numeric(n_features: usize) -> Self {
        let features = vec![FeatureMeta::numeric(); n_features];
        Self::from_features(features)
    }

    /// Number of features in the schema.
    pub fn n_features(&self) -> usize {
        self.features.len()
    }

    /// Get metadata for a feature by index.
    pub fn get(&self, index: usize) -> Option<&FeatureMeta> {
        self.features.get(index)
    }

    /// Get the feature type for a feature by index.
    pub fn feature_type(&self, index: usize) -> FeatureType {
        self.features
            .get(index)
            .map(|m| m.feature_type)
            .unwrap_or(FeatureType::Numeric)
    }

    /// Check if any feature is categorical.
    pub fn has_categorical(&self) -> bool {
        self.features.iter().any(|m| m.feature_type.is_categorical())
    }

    /// Get feature index by name.
    ///
    /// Builds the name index on first call. Returns `None` if no feature
    /// has the given name.
    pub fn feature_index(&mut self, name: &str) -> Option<usize> {
        if self.name_index.is_none() {
            self.build_name_index();
        }
        self.name_index.as_ref().and_then(|idx| idx.get(name).copied())
    }

    /// Build the name index from feature metadata.
    fn build_name_index(&mut self) {
        let mut index = HashMap::new();
        for (i, meta) in self.features.iter().enumerate() {
            if let Some(ref name) = meta.name {
                index.insert(name.clone(), i);
            }
        }
        self.name_index = Some(index);
    }

    /// Get an iterator over feature metadata.
    pub fn iter(&self) -> impl Iterator<Item = &FeatureMeta> {
        self.features.iter()
    }

    /// Get an iterator over (index, metadata) pairs.
    pub fn iter_enumerated(&self) -> impl Iterator<Item = (usize, &FeatureMeta)> {
        self.features.iter().enumerate()
    }

    /// Add a feature to the schema.
    pub fn push(&mut self, meta: FeatureMeta) {
        self.features.push(meta);
        // Invalidate name index
        self.name_index = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_type_default_is_numeric() {
        assert_eq!(FeatureType::default(), FeatureType::Numeric);
    }

    #[test]
    fn feature_type_is_categorical() {
        assert!(FeatureType::Categorical.is_categorical());
        assert!(!FeatureType::Numeric.is_categorical());
    }

    #[test]
    fn feature_meta_numeric() {
        let meta = FeatureMeta::numeric();
        assert_eq!(meta.feature_type, FeatureType::Numeric);
        assert!(meta.name.is_none());

        let meta = FeatureMeta::numeric_named("age");
        assert_eq!(meta.name.as_deref(), Some("age"));
    }

    #[test]
    fn feature_meta_categorical() {
        let meta = FeatureMeta::categorical();
        assert_eq!(meta.feature_type, FeatureType::Categorical);

        let meta = FeatureMeta::categorical_named("color");
        assert_eq!(meta.name.as_deref(), Some("color"));
    }

    #[test]
    fn schema_all_numeric() {
        let schema = DatasetSchema::all_numeric(3);
        assert_eq!(schema.n_features(), 3);
        assert!(!schema.has_categorical());
        assert_eq!(schema.feature_type(0), FeatureType::Numeric);
    }

    #[test]
    fn schema_with_categorical() {
        let schema = DatasetSchema::from_features(vec![
            FeatureMeta::numeric(),
            FeatureMeta::categorical(),
        ]);
        assert!(schema.has_categorical());
        assert_eq!(schema.feature_type(0), FeatureType::Numeric);
        assert_eq!(schema.feature_type(1), FeatureType::Categorical);
    }

    #[test]
    fn schema_feature_index() {
        let mut schema = DatasetSchema::from_features(vec![
            FeatureMeta::numeric_named("a"),
            FeatureMeta::numeric_named("b"),
        ]);
        assert_eq!(schema.feature_index("a"), Some(0));
        assert_eq!(schema.feature_index("b"), Some(1));
        assert_eq!(schema.feature_index("c"), None);
    }

    #[test]
    fn schema_push() {
        let mut schema = DatasetSchema::new();
        schema.push(FeatureMeta::numeric_named("x"));
        schema.push(FeatureMeta::categorical_named("y"));
        assert_eq!(schema.n_features(), 2);
        assert_eq!(schema.feature_index("x"), Some(0));
        assert_eq!(schema.feature_index("y"), Some(1));
    }

    // Verify Send + Sync
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn types_are_send_sync() {
        assert_send_sync::<FeatureType>();
        assert_send_sync::<FeatureMeta>();
        assert_send_sync::<DatasetSchema>();
    }
}
