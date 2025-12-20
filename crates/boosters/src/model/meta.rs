//! Model metadata.
//!
//! Shared metadata types for model introspection.

use serde::{Deserialize, Serialize};

/// Type of machine learning task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TaskKind {
    /// Regression (continuous target).
    #[default]
    Regression,
    /// Binary classification (2 classes).
    BinaryClassification,
    /// Multi-class classification (3+ classes).
    MulticlassClassification {
        /// Number of classes.
        n_classes: usize,
    },
    /// Ranking task.
    Ranking,
}



impl TaskKind {
    /// Returns the number of output groups for this task.
    pub fn n_groups(&self) -> usize {
        match self {
            Self::Regression => 1,
            Self::BinaryClassification => 1,
            Self::MulticlassClassification { n_classes } => *n_classes,
            Self::Ranking => 1,
        }
    }

    /// Returns true if this is a classification task.
    pub fn is_classification(&self) -> bool {
        matches!(
            self,
            Self::BinaryClassification | Self::MulticlassClassification { .. }
        )
    }

    /// Returns true if this is a regression task.
    pub fn is_regression(&self) -> bool {
        matches!(self, Self::Regression)
    }
}

/// Feature type information.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMeta {
    /// Feature names (optional).
    pub feature_names: Option<Vec<String>>,
    /// Feature types (optional).
    pub feature_types: Option<Vec<FeatureType>>,
    /// Number of features.
    pub n_features: usize,
    /// Number of output groups.
    pub n_groups: usize,
    /// Task type.
    pub task: TaskKind,
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
            task: TaskKind::Regression,
            base_scores: vec![0.5],
            ..Default::default()
        }
    }

    /// Create metadata for binary classification.
    pub fn for_binary_classification(n_features: usize) -> Self {
        Self {
            n_features,
            n_groups: 1,
            task: TaskKind::BinaryClassification,
            base_scores: vec![0.0],
            ..Default::default()
        }
    }

    /// Create metadata for multi-class classification.
    pub fn for_multiclass(n_features: usize, n_classes: usize) -> Self {
        Self {
            n_features,
            n_groups: n_classes,
            task: TaskKind::MulticlassClassification { n_classes },
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
    fn task_kind_n_groups() {
        assert_eq!(TaskKind::Regression.n_groups(), 1);
        assert_eq!(TaskKind::BinaryClassification.n_groups(), 1);
        assert_eq!(
            TaskKind::MulticlassClassification { n_classes: 5 }.n_groups(),
            5
        );
    }

    #[test]
    fn task_kind_is_classification() {
        assert!(!TaskKind::Regression.is_classification());
        assert!(TaskKind::BinaryClassification.is_classification());
        assert!(TaskKind::MulticlassClassification { n_classes: 3 }.is_classification());
    }

    #[test]
    fn meta_factories() {
        let reg = ModelMeta::for_regression(10);
        assert_eq!(reg.n_features, 10);
        assert_eq!(reg.n_groups, 1);
        assert!(reg.task.is_regression());

        let bin = ModelMeta::for_binary_classification(5);
        assert!(bin.task.is_classification());

        let multi = ModelMeta::for_multiclass(8, 4);
        assert_eq!(multi.n_groups, 4);
        assert_eq!(multi.base_scores.len(), 4);
    }

    #[test]
    fn meta_serde_roundtrip() {
        let meta = ModelMeta::for_regression(10)
            .with_feature_names(vec!["a".into(), "b".into()])
            .with_best_iteration(42);

        let json = serde_json::to_string(&meta).unwrap();
        let restored: ModelMeta = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.n_features, 10);
        assert_eq!(restored.best_iteration, Some(42));
        assert_eq!(
            restored.feature_names,
            Some(vec!["a".to_string(), "b".to_string()])
        );
    }
}
