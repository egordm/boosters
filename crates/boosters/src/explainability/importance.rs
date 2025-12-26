//! Feature importance computation.
//!
//! Supports multiple importance types for GBDT models.

use std::collections::HashMap;

use crate::repr::gbdt::TreeView;
use crate::repr::gbdt::{Forest, ScalarLeaf};

/// Error type for explainability operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExplainError {
    /// Node statistics (gains/covers) are required but not available.
    MissingNodeStats(&'static str),
    /// Feature statistics (means) are required but not available.
    MissingFeatureStats,
    /// Empty model (no trees).
    EmptyModel,
}

impl std::fmt::Display for ExplainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingNodeStats(stat) => {
                write!(
                    f,
                    "Node {} statistics not available. Train a new model or load one with stats.",
                    stat
                )
            }
            Self::MissingFeatureStats => {
                write!(f, "Feature means not available. Required for linear SHAP.")
            }
            Self::EmptyModel => write!(f, "Model has no trees"),
        }
    }
}

impl std::error::Error for ExplainError {}

/// Type of feature importance to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImportanceType {
    /// Number of times each feature is used in splits.
    #[default]
    Split,
    /// Total gain from splits using each feature.
    Gain,
    /// Average gain per split (Gain / Split count).
    AverageGain,
    /// Total cover (hessian sum) at nodes splitting on each feature.
    Cover,
    /// Average cover per split (Cover / Split count).
    AverageCover,
}

impl ImportanceType {
    /// Check if this importance type requires node statistics.
    pub fn requires_stats(&self) -> bool {
        matches!(
            self,
            Self::Gain | Self::AverageGain | Self::Cover | Self::AverageCover
        )
    }
}

/// Feature importance container with utility methods.
#[derive(Debug, Clone)]
pub struct FeatureImportance {
    /// Raw importance values (one per feature).
    values: Vec<f32>,
    /// Type of importance computed.
    importance_type: ImportanceType,
    /// Optional feature names.
    feature_names: Option<Vec<String>>,
}

impl FeatureImportance {
    /// Create a new feature importance container.
    pub fn new(
        values: Vec<f32>,
        importance_type: ImportanceType,
        feature_names: Option<Vec<String>>,
    ) -> Self {
        Self {
            values,
            importance_type,
            feature_names,
        }
    }

    /// Number of features.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if there are no features.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get importance for a feature by index.
    pub fn get(&self, idx: usize) -> Option<f32> {
        self.values.get(idx).copied()
    }

    /// Get importance for a feature by name.
    pub fn get_by_name(&self, name: &str) -> Option<f32> {
        let names = self.feature_names.as_ref()?;
        let idx = names.iter().position(|n| n == name)?;
        self.get(idx)
    }

    /// Get the importance type.
    pub fn importance_type(&self) -> ImportanceType {
        self.importance_type
    }

    /// Get feature names (if available).
    pub fn feature_names(&self) -> Option<&[String]> {
        self.feature_names.as_deref()
    }

    /// Get raw values slice.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Create a normalized version (sums to 1.0).
    pub fn normalized(&self) -> Self {
        let total: f32 = self.values.iter().sum();
        let values = if total == 0.0 {
            vec![0.0; self.values.len()]
        } else {
            self.values.iter().map(|v| v / total).collect()
        };
        Self {
            values,
            importance_type: self.importance_type,
            feature_names: self.feature_names.clone(),
        }
    }

    /// Get indices sorted by importance (descending).
    pub fn sorted_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.values.len()).collect();
        indices.sort_by(|&a, &b| {
            self.values[b]
                .partial_cmp(&self.values[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }

    /// Get top k features by importance.
    ///
    /// Returns `(index, name, importance)` tuples sorted by importance descending.
    pub fn top_k(&self, k: usize) -> Vec<(usize, Option<String>, f32)> {
        let indices = self.sorted_indices();
        indices
            .into_iter()
            .take(k)
            .map(|idx| {
                let name = self
                    .feature_names
                    .as_ref()
                    .and_then(|names| names.get(idx).cloned());
                (idx, name, self.values[idx])
            })
            .collect()
    }

    /// Convert to a HashMap (feature index → importance).
    pub fn to_map(&self) -> HashMap<usize, f32> {
        self.values.iter().copied().enumerate().collect()
    }

    /// Convert to a HashMap (feature name → importance).
    ///
    /// Returns None if feature names are not available.
    pub fn to_named_map(&self) -> Option<HashMap<String, f32>> {
        let names = self.feature_names.as_ref()?;
        Some(
            names
                .iter()
                .zip(self.values.iter())
                .map(|(name, &val)| (name.clone(), val))
                .collect(),
        )
    }

    /// Iterate over (index, importance) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, f32)> + '_ {
        self.values.iter().copied().enumerate()
    }
}

/// Compute feature importance for a forest.
pub fn compute_forest_importance(
    forest: &Forest<ScalarLeaf>,
    n_features: usize,
    importance_type: ImportanceType,
    feature_names: Option<Vec<String>>,
) -> Result<FeatureImportance, ExplainError> {
    if forest.n_trees() == 0 {
        return Err(ExplainError::EmptyModel);
    }

    // Check if required stats are available
    if importance_type.requires_stats() {
        let first_tree = forest.trees().next().unwrap();
        match importance_type {
            ImportanceType::Gain | ImportanceType::AverageGain => {
                if !first_tree.has_gains() {
                    return Err(ExplainError::MissingNodeStats("gain"));
                }
            }
            ImportanceType::Cover | ImportanceType::AverageCover => {
                if !first_tree.has_covers() {
                    return Err(ExplainError::MissingNodeStats("cover"));
                }
            }
            _ => {}
        }
    }

    let mut split_counts = vec![0u64; n_features];
    let mut values = vec![0.0f32; n_features];

    for tree in forest.trees() {
        // Pre-fetch gains and covers slices for this tree
        let gains_slice = tree.gains();
        let covers_slice = tree.covers();

        for node_idx in 0..tree.n_nodes() as u32 {
            if !tree.is_leaf(node_idx) {
                let feature = tree.split_index(node_idx) as usize;
                if feature < n_features {
                    split_counts[feature] += 1;

                    match importance_type {
                        ImportanceType::Split => {
                            values[feature] += 1.0;
                        }
                        ImportanceType::Gain | ImportanceType::AverageGain => {
                            if let Some(gains) = gains_slice {
                                values[feature] += gains[node_idx as usize];
                            }
                        }
                        ImportanceType::Cover | ImportanceType::AverageCover => {
                            if let Some(covers) = covers_slice {
                                values[feature] += covers[node_idx as usize];
                            }
                        }
                    }
                }
            }
        }
    }

    // For average types, divide by split count
    if importance_type == ImportanceType::AverageGain
        || importance_type == ImportanceType::AverageCover
    {
        for (i, count) in split_counts.iter().enumerate() {
            if *count > 0 {
                values[i] /= *count as f32;
            }
        }
    }

    Ok(FeatureImportance::new(
        values,
        importance_type,
        feature_names,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_forest_with_stats() -> Forest<ScalarLeaf> {
        let tree1 = crate::scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => num(1, 0.3, R) -> 3, 4,
            3 => leaf(2.0),
            4 => leaf(3.0),
        };
        let tree1 = tree1
            .with_gains(vec![10.0, 0.0, 5.0, 0.0, 0.0])
            .with_covers(vec![100.0, 40.0, 60.0, 30.0, 30.0]);

        let tree2 = crate::scalar_tree! {
            0 => num(0, 0.7, L) -> 1, 2,
            1 => leaf(0.5),
            2 => leaf(1.5),
        };
        let tree2 = tree2
            .with_gains(vec![8.0, 0.0, 0.0])
            .with_covers(vec![100.0, 60.0, 40.0]);

        let mut forest = Forest::for_regression().with_base_score(vec![0.0]);
        forest.push_tree(tree1, 0);
        forest.push_tree(tree2, 0);
        forest
    }

    #[test]
    fn split_importance() {
        let forest = make_forest_with_stats();
        let imp = compute_forest_importance(&forest, 2, ImportanceType::Split, None).unwrap();

        // feature 0 used 2 times (tree1 root, tree2 root)
        // feature 1 used 1 time (tree1 node 2)
        assert_eq!(imp.values(), &[2.0, 1.0]);
    }

    #[test]
    fn gain_importance() {
        let forest = make_forest_with_stats();
        let imp = compute_forest_importance(&forest, 2, ImportanceType::Gain, None).unwrap();

        // feature 0: 10.0 + 8.0 = 18.0
        // feature 1: 5.0
        assert_eq!(imp.values(), &[18.0, 5.0]);
    }

    #[test]
    fn average_gain_importance() {
        let forest = make_forest_with_stats();
        let imp = compute_forest_importance(&forest, 2, ImportanceType::AverageGain, None).unwrap();

        // feature 0: (10.0 + 8.0) / 2 = 9.0
        // feature 1: 5.0 / 1 = 5.0
        assert_eq!(imp.values(), &[9.0, 5.0]);
    }

    #[test]
    fn cover_importance() {
        let forest = make_forest_with_stats();
        let imp = compute_forest_importance(&forest, 2, ImportanceType::Cover, None).unwrap();

        // feature 0: 100.0 + 100.0 = 200.0
        // feature 1: 60.0
        assert_eq!(imp.values(), &[200.0, 60.0]);
    }

    #[test]
    fn missing_stats_error() {
        // Forest without stats
        let tree = crate::scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => leaf(2.0),
        };
        let mut forest = Forest::for_regression().with_base_score(vec![0.0]);
        forest.push_tree(tree, 0);

        // Split works (no stats needed)
        assert!(compute_forest_importance(&forest, 2, ImportanceType::Split, None).is_ok());

        // Gain fails (needs stats)
        assert!(matches!(
            compute_forest_importance(&forest, 2, ImportanceType::Gain, None),
            Err(ExplainError::MissingNodeStats("gain"))
        ));

        // Cover fails (needs stats)
        assert!(matches!(
            compute_forest_importance(&forest, 2, ImportanceType::Cover, None),
            Err(ExplainError::MissingNodeStats("cover"))
        ));
    }

    #[test]
    fn normalized_importance() {
        let imp = FeatureImportance::new(vec![10.0, 20.0, 30.0], ImportanceType::Split, None);
        let norm = imp.normalized();

        let sum: f32 = norm.values().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        assert!((norm.get(0).unwrap() - (10.0 / 60.0)).abs() < 1e-5);
        assert!((norm.get(1).unwrap() - (20.0 / 60.0)).abs() < 1e-5);
        assert!((norm.get(2).unwrap() - (30.0 / 60.0)).abs() < 1e-5);
    }

    #[test]
    fn top_k() {
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let imp =
            FeatureImportance::new(vec![10.0, 30.0, 20.0], ImportanceType::Split, Some(names));

        let top2 = imp.top_k(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, 1); // "b" with 30.0
        assert_eq!(top2[0].1, Some("b".to_string()));
        assert_eq!(top2[0].2, 30.0);
        assert_eq!(top2[1].0, 2); // "c" with 20.0
    }

    #[test]
    fn get_by_name() {
        let names = vec!["alpha".to_string(), "beta".to_string()];
        let imp = FeatureImportance::new(vec![1.0, 2.0], ImportanceType::Split, Some(names));

        assert_eq!(imp.get_by_name("alpha"), Some(1.0));
        assert_eq!(imp.get_by_name("beta"), Some(2.0));
        assert_eq!(imp.get_by_name("gamma"), None);
    }
}
