//! Forest (tree ensemble) for gradient boosting.

use super::node::Tree;

/// Collection of trees forming an ensemble (forest).
#[derive(Clone, Debug)]
pub struct Forest {
    /// All trees, grouped by output.
    trees: Vec<Tree>,
    /// Base score per output (initial prediction before any trees).
    base_scores: Vec<f32>,
    /// Number of outputs (1 for regression/binary, K for K-class).
    n_outputs: usize,
}

impl Forest {
    /// Create a new forest.
    pub fn new(n_outputs: usize, base_scores: Vec<f32>) -> Self {
        assert_eq!(base_scores.len(), n_outputs);
        Self {
            trees: Vec::new(),
            base_scores,
            n_outputs,
        }
    }

    /// Create forest for single-output model.
    pub fn single_output(base_score: f32) -> Self {
        Self::new(1, vec![base_score])
    }

    /// Add a tree for a specific output.
    pub fn add_tree(&mut self, tree: Tree, _output: usize) {
        self.trees.push(tree);
    }

    /// Get all trees.
    #[inline]
    pub fn trees(&self) -> &[Tree] {
        &self.trees
    }

    /// Get trees for a specific output.
    pub fn trees_for_output(&self, output: usize) -> impl Iterator<Item = &Tree> {
        let n_outputs = self.n_outputs;
        self.trees
            .iter()
            .enumerate()
            .filter(move |(i, _)| i % n_outputs == output)
            .map(|(_, t)| t)
    }

    /// Total number of trees.
    #[inline]
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Number of outputs.
    #[inline]
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// Base scores.
    #[inline]
    pub fn base_scores(&self) -> &[f32] {
        &self.base_scores
    }

    /// Number of trees per output.
    pub fn trees_per_output(&self) -> usize {
        if self.n_outputs == 0 {
            0
        } else {
            self.trees.len() / self.n_outputs
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::builder::TreeBuilder;
    use super::*;

    #[test]
    fn test_forest() {
        let mut forest = Forest::single_output(0.5);

        // Add a simple tree
        let mut builder = TreeBuilder::new();
        builder.init_root();
        builder.make_leaf(0, 0.1);
        forest.add_tree(builder.finish(), 0);

        assert_eq!(forest.n_trees(), 1);
        assert_eq!(forest.n_outputs(), 1);
        assert_eq!(forest.base_scores(), &[0.5]);
        assert_eq!(forest.trees_per_output(), 1);
    }
}
