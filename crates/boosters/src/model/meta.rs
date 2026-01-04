//! Model metadata.
//!
//! Shared metadata types for model introspection.

/// Feature type information.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum FeatureType {
    /// Numeric feature.
    #[default]
    Numeric,
    /// Categorical feature.
    Categorical {
        /// Number of categories (if known).
        n_categories: Option<u32>,
    },
}

/// Shared metadata for all model types.
///
/// Contains introspection data about the model's structure and training context.
#[derive(Debug, Clone, Default)]
pub struct ModelMeta {
    /// Feature names (optional).
    pub feature_names: Option<Vec<String>>,
    /// Feature types (optional).
    pub feature_types: Option<Vec<FeatureType>>,
    /// Number of features.
    pub n_features: usize,
    /// Number of output groups.
    pub n_groups: usize,
    /// Best iteration (from early stopping).
    pub best_iteration: Option<usize>,
    /// Base scores (one per group).
    pub base_scores: Vec<f32>,
}

impl ModelMeta {
    /// Create metadata for a regression task.
    pub fn for_regression(n_features: usize) -> Self {
        Self {
            n_features,
            n_groups: 1,
            base_scores: vec![0.5],
            ..Default::default()
        }
    }

    /// Create metadata for binary classification.
    pub fn for_binary_classification(n_features: usize) -> Self {
        Self {
            n_features,
            n_groups: 1,
            base_scores: vec![0.0],
            ..Default::default()
        }
    }

    /// Create metadata for multi-class classification.
    pub fn for_multiclass(n_features: usize, n_classes: usize) -> Self {
        Self {
            n_features,
            n_groups: n_classes,
            base_scores: vec![0.0; n_classes],
            ..Default::default()
        }
    }

    /// Set feature names.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Set base scores.
    pub fn with_base_scores(mut self, scores: Vec<f32>) -> Self {
        self.base_scores = scores;
        self
    }

    /// Set best iteration.
    pub fn with_best_iteration(mut self, iter: usize) -> Self {
        self.best_iteration = Some(iter);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_factories() {
        let reg = ModelMeta::for_regression(10);
        assert_eq!(reg.n_features, 10);
        assert_eq!(reg.n_groups, 1);

        let bin = ModelMeta::for_binary_classification(5);
        assert_eq!(bin.n_groups, 1);

        let multi = ModelMeta::for_multiclass(8, 4);
        assert_eq!(multi.n_groups, 4);
        assert_eq!(multi.base_scores.len(), 4);
    }
}
