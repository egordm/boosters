//! Canonical forest representation (collection of trees).

use super::{LeafValue, ScalarLeaf, Tree, TreeValidationError};

/// Structural validation errors for [`Forest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForestValidationError {
    BaseScoreLenMismatch {
        n_groups: u32,
        len: usize,
    },
    TreeGroupsLenMismatch {
        n_trees: usize,
        len: usize,
    },
    TreeGroupOutOfRange {
        tree_idx: usize,
        group: u32,
        n_groups: u32,
    },
    InvalidTree {
        tree_idx: usize,
        error: TreeValidationError,
    },
}

/// Forest of decision trees.
///
/// Stores multiple trees with their group assignments for multi-class support.
#[derive(Debug, Clone)]
pub struct Forest<L: LeafValue = ScalarLeaf> {
    trees: Vec<Tree<L>>,
    tree_groups: Vec<u32>,
    n_groups: u32,
    base_score: Vec<f32>,
}

impl<L: LeafValue> Forest<L> {
    /// Create a new forest with the given number of groups.
    pub fn new(n_groups: u32) -> Self {
        Self {
            trees: Vec::new(),
            tree_groups: Vec::new(),
            n_groups,
            base_score: vec![0.0; n_groups as usize],
        }
    }

    /// Create a forest for regression (single output group).
    pub fn for_regression() -> Self {
        Self::new(1)
    }

    /// Set the base score for all groups.
    pub fn with_base_score(mut self, base_score: Vec<f32>) -> Self {
        debug_assert_eq!(base_score.len(), self.n_groups as usize);
        self.base_score = base_score;
        self
    }

    /// Add a tree to the forest.
    pub fn push_tree(&mut self, tree: Tree<L>, group: u32) {
        debug_assert!(group < self.n_groups, "group out of range");
        self.trees.push(tree);
        self.tree_groups.push(group);
    }

    /// Number of trees.
    #[inline]
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Number of output groups.
    #[inline]
    pub fn n_groups(&self) -> u32 {
        self.n_groups
    }

    /// Get the base score for each group.
    #[inline]
    pub fn base_score(&self) -> &[f32] {
        &self.base_score
    }

    /// Get a reference to a specific tree.
    #[inline]
    pub fn tree(&self, idx: usize) -> &Tree<L> {
        &self.trees[idx]
    }

    /// Get all tree group assignments as a slice.
    #[inline]
    pub fn tree_groups(&self) -> &[u32] {
        &self.tree_groups
    }

    /// Iterate over trees.
    pub fn trees(&self) -> impl Iterator<Item = &Tree<L>> {
        self.trees.iter()
    }

    /// Iterate over trees with their group assignments.
    pub fn trees_with_groups(&self) -> impl Iterator<Item = (&Tree<L>, u32)> {
        self.trees
            .iter()
            .zip(self.tree_groups.iter())
            .map(|(t, &g)| (t, g))
    }

    /// Validate structural invariants for this forest (trees, group assignments, base score).
    ///
    /// Intended for debug checks and tests (e.g., model conversion invariants).
    pub fn validate(&self) -> Result<(), ForestValidationError> {
        if self.base_score.len() != self.n_groups as usize {
            return Err(ForestValidationError::BaseScoreLenMismatch {
                n_groups: self.n_groups,
                len: self.base_score.len(),
            });
        }
        if self.tree_groups.len() != self.trees.len() {
            return Err(ForestValidationError::TreeGroupsLenMismatch {
                n_trees: self.trees.len(),
                len: self.tree_groups.len(),
            });
        }

        for (i, &g) in self.tree_groups.iter().enumerate() {
            if g >= self.n_groups {
                return Err(ForestValidationError::TreeGroupOutOfRange {
                    tree_idx: i,
                    group: g,
                    n_groups: self.n_groups,
                });
            }
        }

        for (i, tree) in self.trees.iter().enumerate() {
            tree.validate()
                .map_err(|e| ForestValidationError::InvalidTree {
                    tree_idx: i,
                    error: e,
                })?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::gbdt::SimplePredictor;
    use crate::repr::gbdt::ScalarLeaf;

    fn build_simple_tree(left_val: f32, right_val: f32, threshold: f32) -> Tree<ScalarLeaf> {
        crate::scalar_tree! {
            0 => num(0, threshold, L) -> 1, 2,
            1 => leaf(left_val),
            2 => leaf(right_val),
        }
    }

    #[test]
    fn forest_single_tree_regression() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = SimplePredictor::new(&forest);
        let mut output = vec![0.0; 1];

        predictor.predict_row_into(ndarray::array![0.3f32].view(), None, &mut output);
        assert_eq!(output, vec![1.0]);

        predictor.predict_row_into(ndarray::array![0.7f32].view(), None, &mut output);
        assert_eq!(output, vec![2.0]);
    }
}
