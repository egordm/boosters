//! Leaf linear trainer for fitting linear models at tree leaves.
//!
//! Orchestrates the full training flow:
//! 1. For each leaf, select features (path or global based on config)
//! 2. Gather feature values into column-major buffer
//! 3. Fit weighted least squares using coordinate descent
//! 4. Return fitted coefficients
//!
//! See RFC-0010 for design rationale.

use crate::data::Dataset;
use crate::repr::gbdt::{MutableTree, ScalarLeaf, SplitType, TreeView};
use crate::training::gbdt::partition::RowPartitioner;
use crate::training::{Gradients, GradsTuple};

use super::{
    LeafFeatureBuffer, LinearFeatureSelection, LinearLeafConfig, WeightedLeastSquaresSolver,
};

/// Result of fitting a linear model to a leaf.
#[derive(Debug, Clone)]
pub struct FittedLeaf {
    /// Node ID of the leaf
    pub node_id: u32,
    /// Feature indices used (in order)
    pub features: Vec<u32>,
    /// Intercept term (adjustment to base leaf value)
    pub intercept: f32,
    /// Coefficients for each feature (same order as features)
    pub coefficients: Vec<f32>,
}

impl FittedLeaf {
    /// Create a new fitted leaf result.
    pub fn new(node_id: u32, features: Vec<u32>, intercept: f32, coefficients: Vec<f32>) -> Self {
        debug_assert_eq!(features.len(), coefficients.len());
        Self {
            node_id,
            features,
            intercept,
            coefficients,
        }
    }

    /// Check if any coefficients are non-zero.
    pub fn has_coefficients(&self) -> bool {
        !self.coefficients.is_empty()
    }
}

/// Orchestrates linear fitting for all leaves in a tree.
///
/// Pre-allocates buffers and reuses them for each leaf.
pub struct LeafLinearTrainer {
    config: LinearLeafConfig,
    /// Pre-allocated feature buffer (reused per leaf)
    feature_buffer: LeafFeatureBuffer,
    /// Pre-allocated solver (reused per leaf)
    solver: WeightedLeastSquaresSolver,
    /// Pre-allocated gradient/hessian buffer (reused per leaf)
    grad_hess_buffer: Vec<GradsTuple>,
    /// Cached global features for GlobalFeatures mode (computed once per tree)
    cached_global_features: Vec<u32>,
}

impl LeafLinearTrainer {
    /// Create a new trainer with the given config and maximum leaf size.
    ///
    /// # Arguments
    /// * `config` - Linear leaf configuration
    /// * `max_leaf_samples` - Maximum expected samples in a leaf (for buffer sizing)
    pub fn new(config: LinearLeafConfig, max_leaf_samples: usize) -> Self {
        Self {
            feature_buffer: LeafFeatureBuffer::new(max_leaf_samples, config.max_features),
            solver: WeightedLeastSquaresSolver::new(config.max_features),
            grad_hess_buffer: vec![GradsTuple::default(); max_leaf_samples],
            cached_global_features: Vec::new(),
            config,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &LinearLeafConfig {
        &self.config
    }

    /// Compute global features from a tree based on split frequency.
    ///
    /// Returns the top-k most frequently used numeric features across all splits.
    /// This is used for GlobalFeatures mode where all leaves use the same features.
    fn compute_global_features<T: TreeView>(&self, tree: &T) -> Vec<u32> {
        use std::collections::HashMap;

        let mut feature_counts: HashMap<u32, u32> = HashMap::new();

        // Walk all nodes and count numeric split features
        for node_id in 0..tree.n_nodes() {
            if !tree.is_leaf(node_id as u32)
                && tree.split_type(node_id as u32) == SplitType::Numeric
            {
                let feat = tree.split_index(node_id as u32);
                *feature_counts.entry(feat).or_insert(0) += 1;
            }
        }

        // Sort by count (descending) and take top-k
        let mut features: Vec<(u32, u32)> = feature_counts.into_iter().collect();
        features.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        features
            .into_iter()
            .take(self.config.max_features)
            .map(|(f, _)| f)
            .collect()
    }

    /// Train linear models for all leaves in a tree.
    ///
    /// Returns a vector of fitted leaves. Leaves that are skipped (too few samples,
    /// no features, or failed convergence) are not included.
    ///
    /// # Arguments
    /// * `tree` - The tree being trained (implements TreeView for path traversal)
    /// * `data` - Dataset containing raw feature values
    /// * `partitioner` - Row partitioner with leaf assignments
    /// * `leaf_node_mapping` - Mapping from tree node IDs to partitioner node IDs
    /// * `gradients` - Gradient/hessian values
    /// * `output` - Output index for multi-output (usually 0)
    /// * `learning_rate` - Learning rate to apply to coefficients
    ///
    /// # Example
    ///
    /// ```ignore
    /// // During GBDT training:
    /// let mapping = grower.leaf_node_mapping();
    /// let fitted = trainer.train(&tree, &dataset, partitioner, mapping, gradients, 0, 0.1);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn train(
        &mut self,
        tree: &MutableTree<ScalarLeaf>,
        data: &Dataset,
        partitioner: &RowPartitioner,
        leaf_node_mapping: &[(u32, u32)],
        gradients: &Gradients,
        output: usize,
        learning_rate: f32,
    ) -> Vec<FittedLeaf> {
        // Compute global features once if using GlobalFeatures mode
        let use_global = matches!(
            self.config.feature_selection,
            LinearFeatureSelection::GlobalFeatures
        );
        if use_global {
            self.cached_global_features = self.compute_global_features(tree);
            if self.cached_global_features.is_empty() {
                return Vec::new();
            }
        }

        let mut results = Vec::new();

        for &(tree_node, partitioner_node) in leaf_node_mapping {
            // Get samples in this leaf using partitioner node ID
            let rows = partitioner.get_leaf_indices(partitioner_node);

            if rows.len() < self.config.min_samples {
                continue;
            }

            // Select features based on mode
            let features = if use_global {
                self.cached_global_features.clone()
            } else {
                self.select_path_features(tree, tree_node)
            };

            if features.is_empty() {
                continue;
            }

            // Fit linear model - uses tree node ID for the result
            if let Some(fitted) = self.fit_leaf(
                tree_node,
                &features,
                rows,
                data,
                gradients,
                output,
                learning_rate,
            ) {
                results.push(fitted);
            }
        }

        results
    }

    /// Select numeric features on the path from root to leaf.
    ///
    /// Walks from root to the given leaf, collecting feature indices
    /// from numeric splits only. Categorical splits are skipped.
    fn select_path_features<T: TreeView>(&self, tree: &T, leaf_node: u32) -> Vec<u32> {
        let mut features = Vec::new();
        let mut visited = std::collections::HashSet::new();

        // Walk from root to leaf
        Self::collect_path_features_recursive(tree, 0, leaf_node, &mut features, &mut visited);

        // Limit to max_features
        features.truncate(self.config.max_features);
        features
    }

    /// Recursively collect path features while walking to target leaf.
    fn collect_path_features_recursive<T: TreeView>(
        tree: &T,
        current: u32,
        target_leaf: u32,
        features: &mut Vec<u32>,
        visited: &mut std::collections::HashSet<u32>,
    ) -> bool {
        if current == target_leaf {
            return true;
        }

        if tree.is_leaf(current) {
            return false;
        }

        // Check left subtree
        let left = tree.left_child(current);
        let right = tree.right_child(current);

        // Try left path
        if Self::collect_path_features_recursive(tree, left, target_leaf, features, visited) {
            // Found target in left subtree - add this node's feature if numeric
            if tree.split_type(current) == SplitType::Numeric {
                let feat = tree.split_index(current);
                if visited.insert(feat) {
                    features.push(feat);
                }
            }
            return true;
        }

        // Try right path
        if Self::collect_path_features_recursive(tree, right, target_leaf, features, visited) {
            if tree.split_type(current) == SplitType::Numeric {
                let feat = tree.split_index(current);
                if visited.insert(feat) {
                    features.push(feat);
                }
            }
            return true;
        }

        false
    }

    /// Fit a linear model for a single leaf.
    #[allow(clippy::too_many_arguments)]
    fn fit_leaf(
        &mut self,
        node_id: u32,
        features: &[u32],
        rows: &[u32],
        data: &Dataset,
        gradients: &Gradients,
        output: usize,
        learning_rate: f32,
    ) -> Option<FittedLeaf> {
        let n_features = features.len();
        let n_rows = rows.len();

        // Gather features using Dataset
        self.feature_buffer.gather(rows, data, features);

        // Reset solver and accumulate
        self.solver.reset(n_features);

        // Collect gradients and hessians into pre-allocated buffer
        for (i, &row) in rows.iter().enumerate() {
            let (g, h) = gradients.get(row as usize, output);
            self.grad_hess_buffer[i] = GradsTuple { grad: g, hess: h };
        }
        let grad_hess = &self.grad_hess_buffer[..n_rows];

        // Accumulate intercept
        self.solver.accumulate_intercept(grad_hess);

        // Accumulate each feature column
        for feat_idx in 0..n_features {
            let feat_values = self.feature_buffer.feature_slice(feat_idx);
            self.solver
                .accumulate_column(feat_idx, feat_values, grad_hess);
        }

        // Accumulate cross-terms
        for i in 0..n_features {
            let col_i = self.feature_buffer.feature_slice(i);
            for j in (i + 1)..n_features {
                let col_j = self.feature_buffer.feature_slice(j);
                self.solver
                    .accumulate_cross_term(i, j, col_i, col_j, grad_hess);
            }
        }

        // Add regularization
        self.solver.add_regularization(self.config.lambda as f64);

        // Solve using coordinate descent.
        // We use the coefficients even if not fully converged, matching LightGBM's
        // linear tree behavior. The tolerance enables early stopping optimization.
        self.solver
            .solve_cd(self.config.max_iterations, self.config.tolerance);

        // Get coefficients
        let (intercept, coefs) = self.solver.coefficients();

        // Prune near-zero coefficients
        let threshold = self.config.coefficient_threshold as f64;
        let (pruned_features, pruned_coefs): (Vec<u32>, Vec<f32>) = features
            .iter()
            .zip(coefs.iter())
            .filter(|(_, c)| c.abs() > threshold)
            .map(|(f, c)| (*f, (*c * learning_rate as f64) as f32))
            .unzip();

        if pruned_features.is_empty() && intercept.abs() < threshold {
            return None; // All coefficients pruned
        }

        Some(FittedLeaf::new(
            node_id,
            pruned_features,
            (intercept * learning_rate as f64) as f32,
            pruned_coefs,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repr::gbdt::{MutableTree, ScalarLeaf};

    /// Create a simple 3-node tree: root splits on feature 0, two leaves
    fn make_simple_tree() -> MutableTree<ScalarLeaf> {
        let mut tree = MutableTree::new();
        tree.init_root();
        // Root at node 0 splits on feature 0 at threshold 0.5, default_left=true
        // Returns (left_child, right_child) = (1, 2)
        let (left, right) = tree.apply_numeric_split(0, 0, 0.5, true);
        // Mark children as leaves
        tree.make_leaf(left, ScalarLeaf(1.0));
        tree.make_leaf(right, ScalarLeaf(2.0));
        tree
    }

    /// Create a deeper tree: root splits on f0, left child splits on f1
    fn make_deep_tree() -> MutableTree<ScalarLeaf> {
        let mut tree = MutableTree::new();
        tree.init_root();
        // Root splits on feature 0
        let (left, right) = tree.apply_numeric_split(0, 0, 0.5, true);
        tree.make_leaf(right, ScalarLeaf(2.0)); // Right is a leaf

        // Left child splits on feature 1
        let (ll, lr) = tree.apply_numeric_split(left, 1, 0.3, true);
        tree.make_leaf(ll, ScalarLeaf(3.0));
        tree.make_leaf(lr, ScalarLeaf(4.0));
        tree
    }

    #[test]
    fn test_select_path_features_simple() {
        let tree = make_simple_tree();
        let frozen = tree.clone().freeze();

        let config = LinearLeafConfig::default();
        let trainer = LeafLinearTrainer::new(config, 100);

        // Leaf 1 (left child of root) - path uses feature 0
        let features = trainer.select_path_features(&frozen, 1);
        assert_eq!(
            features,
            vec![0],
            "Left leaf should see feature 0 from root"
        );

        // Leaf 2 (right child of root) - path uses feature 0
        let features = trainer.select_path_features(&frozen, 2);
        assert_eq!(
            features,
            vec![0],
            "Right leaf should see feature 0 from root"
        );
    }

    #[test]
    fn test_select_path_features_deep() {
        let tree = make_deep_tree();
        let frozen = tree.clone().freeze();

        let config = LinearLeafConfig::default();
        let trainer = LeafLinearTrainer::new(config, 100);

        // Tree structure:
        //        0 (f0)
        //       / \
        //      1   2
        //     (f1)
        //    /   \
        //   3     4

        // Leaf 3 path: 0 -> 1 -> 3, sees f0 and f1
        let features = trainer.select_path_features(&frozen, 3);
        assert!(features.contains(&0), "Deep leaf should see f0 from root");
        assert!(features.contains(&1), "Deep leaf should see f1 from parent");
        assert_eq!(features.len(), 2, "Should have exactly 2 path features");

        // Leaf 4 path: 0 -> 1 -> 4, sees f0 and f1
        let features = trainer.select_path_features(&frozen, 4);
        assert!(features.contains(&0), "Deep leaf should see f0 from root");
        assert!(features.contains(&1), "Deep leaf should see f1 from parent");
        assert_eq!(features.len(), 2, "Should have exactly 2 path features");

        // Leaf 2 path: 0 -> 2, sees only f0
        let features = trainer.select_path_features(&frozen, 2);
        assert_eq!(features, vec![0], "Shallow leaf should only see f0");
    }

    #[test]
    fn test_select_path_features_max_limit() {
        // Create a deeper tree with 5 levels
        let mut tree: MutableTree<ScalarLeaf> = MutableTree::new();
        tree.init_root();

        // Split root on f0
        let (l1, r1) = tree.apply_numeric_split(0, 0, 0.5, true);
        tree.make_leaf(r1, ScalarLeaf(0.0));

        // Split left child on f1
        let (l2, r2) = tree.apply_numeric_split(l1, 1, 0.5, true);
        tree.make_leaf(r2, ScalarLeaf(0.0));

        // Split that child on f2
        let (l3, r3) = tree.apply_numeric_split(l2, 2, 0.5, true);
        tree.make_leaf(r3, ScalarLeaf(0.0));

        // Split that child on f3
        let (l4, r4) = tree.apply_numeric_split(l3, 3, 0.5, true);
        tree.make_leaf(r4, ScalarLeaf(0.0));

        // Split that child on f4 (different from f2)
        let (l5, r5) = tree.apply_numeric_split(l4, 4, 0.3, true);
        tree.make_leaf(l5, ScalarLeaf(0.0));
        tree.make_leaf(r5, ScalarLeaf(0.0));

        let frozen = tree.clone().freeze();

        // With max_features=3, should only get 3 features
        let config = LinearLeafConfig::builder().max_features(3).build();
        let trainer = LeafLinearTrainer::new(config, 100);

        // Path to l5 has f0, f1, f2, f3, f4 = 5 unique features, but limited to 3
        let features = trainer.select_path_features(&frozen, l5);
        assert!(
            features.len() <= 3,
            "Should respect max_features limit, got {}",
            features.len()
        );

        // Path to r4 has f0, f1, f2, f3 = 4 features, limited to 3
        let features_r4 = trainer.select_path_features(&frozen, r4);
        assert!(features_r4.len() <= 3, "Should respect max_features limit");
    }

    #[test]
    fn test_fitted_leaf_creation() {
        let fitted = FittedLeaf::new(5, vec![0, 2], 1.5, vec![0.5, -0.3]);

        assert_eq!(fitted.node_id, 5);
        assert_eq!(fitted.features, vec![0, 2]);
        assert_eq!(fitted.intercept, 1.5);
        assert_eq!(fitted.coefficients, vec![0.5, -0.3]);
        assert!(fitted.has_coefficients(), "Should have coefficients");
    }

    #[test]
    fn test_fitted_leaf_intercept_only() {
        let fitted = FittedLeaf::new(3, vec![], 2.0, vec![]);

        assert_eq!(fitted.intercept, 2.0);
        assert!(fitted.features.is_empty());
        assert!(!fitted.has_coefficients(), "Should not have coefficients");
    }

    #[test]
    fn test_trainer_new() {
        let config = LinearLeafConfig::builder()
            .min_samples(20)
            .max_features(5)
            .build();

        let trainer = LeafLinearTrainer::new(config, 100);

        assert_eq!(trainer.config().min_samples, 20);
        assert_eq!(trainer.config().max_features, 5);
    }

    /// Test that LeafLinearTrainer can be instantiated with the buffer module.
    /// The actual fitting is tested in integration tests with full tree grower.
    #[test]
    fn test_feature_buffer_gather_with_dataset() {
        use crate::data::Dataset;
        use ndarray::Array2;

        // Create dataset with feature-major layout [n_features, n_samples]
        let data: Vec<f32> = vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0];
        let arr = Array2::from_shape_vec((1, 10), data).unwrap();
        let dataset = Dataset::from_array(arr.view(), None, None);

        // Create feature buffer
        let mut buffer = super::LeafFeatureBuffer::new(10, 5);

        // Gather features for a subset of rows
        let rows = [0u32, 2, 4, 6, 8];
        let features = [0u32];
        buffer.gather(&rows, &dataset, &features);

        // Verify the gathered values are raw (not binned)
        let gathered = buffer.feature_slice(0);
        assert!(
            (gathered[0] - 1.1).abs() < 0.01,
            "Expected 1.1, got {}",
            gathered[0]
        );
        assert!(
            (gathered[1] - 3.3).abs() < 0.01,
            "Expected 3.3, got {}",
            gathered[1]
        );
        assert!(
            (gathered[2] - 5.5).abs() < 0.01,
            "Expected 5.5, got {}",
            gathered[2]
        );
        assert!(
            (gathered[3] - 7.7).abs() < 0.01,
            "Expected 7.7, got {}",
            gathered[3]
        );
        assert!(
            (gathered[4] - 9.9).abs() < 0.01,
            "Expected 9.9, got {}",
            gathered[4]
        );
    }
}
