//! Block-based tree traversal for improved cache efficiency.
//!
//! This module implements block-based traversal where multiple rows are processed
//! through trees together. This improves cache locality because:
//!
//! 1. Tree nodes stay in L1/L2 cache while processing multiple rows
//! 2. Feature data is accessed in a more predictable pattern
//! 3. Output accumulation is batched
//!
//! See [design/concepts/block_based_traversal.md] for theory.
//!
//! # Block Size
//!
//! The default block size is 64 rows, matching XGBoost. This balances:
//! - Cache utilization (larger blocks keep tree data hot)
//! - Memory overhead (per-block buffers)
//! - Parallelism granularity

use crate::data::DataMatrix;
use crate::forest::SoAForest;
use crate::trees::{LeafValue, ScalarLeaf};

use super::output::PredictionOutput;
use super::visitor::ScalarVisitor;
use super::TreeVisitor;

/// Default block size for batch processing (matches XGBoost).
pub const DEFAULT_BLOCK_SIZE: usize = 64;

/// Configuration for block-based prediction.
#[derive(Debug, Clone)]
pub struct BlockConfig {
    /// Number of rows to process together.
    pub block_size: usize,
}

impl Default for BlockConfig {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
        }
    }
}

/// Block-based predictor for improved batch performance.
///
/// Processes rows in blocks to improve cache locality. Each block of rows
/// is processed through all trees before moving to the next block.
///
/// # When to Use
///
/// - **Use `BlockPredictor`**: When predicting on 100+ rows at a time
/// - **Use `Predictor`**: For single-row or small batch predictions
///
/// # Example
///
/// ```ignore
/// use booste_rs::predict::BlockPredictor;
/// use booste_rs::data::DenseMatrix;
///
/// let predictor = BlockPredictor::new(&forest);
/// let features = DenseMatrix::from_vec(data, 1000, 50);
/// let output = predictor.predict(&features);
/// ```
#[derive(Debug, Clone)]
pub struct BlockPredictor<'f, L: LeafValue = ScalarLeaf> {
    forest: &'f SoAForest<L>,
    config: BlockConfig,
}

impl<'f, L: LeafValue> BlockPredictor<'f, L> {
    /// Create a new block predictor with default configuration.
    pub fn new(forest: &'f SoAForest<L>) -> Self {
        Self {
            forest,
            config: BlockConfig::default(),
        }
    }

    /// Create a new block predictor with custom configuration.
    pub fn with_config(forest: &'f SoAForest<L>, config: BlockConfig) -> Self {
        Self { forest, config }
    }

    /// Get the block size.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// Get a reference to the underlying forest.
    #[inline]
    pub fn forest(&self) -> &SoAForest<L> {
        self.forest
    }

    /// Number of output groups.
    #[inline]
    pub fn num_groups(&self) -> usize {
        self.forest.num_groups() as usize
    }
}

impl<'f> BlockPredictor<'f, ScalarLeaf> {
    /// Predict for a batch of features using block-based traversal.
    ///
    /// Returns a [`PredictionOutput`] with shape `(num_rows, num_groups)`.
    pub fn predict<M: DataMatrix<Element = f32>>(&self, features: &M) -> PredictionOutput {
        let num_rows = features.num_rows();
        let num_groups = self.num_groups();
        let num_features = features.num_features();
        let block_size = self.config.block_size;

        let mut output = PredictionOutput::zeros(num_rows, num_groups);
        let visitor = ScalarVisitor;

        // Initialize with base scores
        let base_score = self.forest.base_score();
        for row_idx in 0..num_rows {
            output.row_mut(row_idx).copy_from_slice(base_score);
        }

        // Pre-allocate feature buffer for the block
        let actual_block_size = block_size.min(num_rows);
        let mut feature_buffer: Vec<Vec<f32>> = (0..actual_block_size)
            .map(|_| vec![f32::NAN; num_features])
            .collect();

        // Process in blocks
        for block_start in (0..num_rows).step_by(block_size) {
            let block_end = (block_start + block_size).min(num_rows);
            let current_block_size = block_end - block_start;

            // Load features for this block
            for (i, row_idx) in (block_start..block_end).enumerate() {
                features.copy_row(row_idx, &mut feature_buffer[i]);
            }

            // Process all trees for this block
            // Key optimization: tree data stays in cache while processing multiple rows
            for (tree, group) in self.forest.trees_with_groups() {
                for i in 0..current_block_size {
                    let row_idx = block_start + i;
                    let leaf_value = visitor.visit_tree(&tree, &feature_buffer[i]);
                    output.row_mut(row_idx)[group as usize] += leaf_value;
                }
            }
        }

        output
    }

    /// Predict with per-tree weights (for DART) using block-based traversal.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != forest.num_trees()`.
    pub fn predict_weighted<M: DataMatrix<Element = f32>>(
        &self,
        features: &M,
        weights: &[f32],
    ) -> PredictionOutput {
        assert_eq!(
            weights.len(),
            self.forest.num_trees(),
            "weights length must match number of trees"
        );

        let num_rows = features.num_rows();
        let num_groups = self.num_groups();
        let num_features = features.num_features();
        let block_size = self.config.block_size;

        let mut output = PredictionOutput::zeros(num_rows, num_groups);
        let visitor = ScalarVisitor;

        // Initialize with base scores
        let base_score = self.forest.base_score();
        for row_idx in 0..num_rows {
            output.row_mut(row_idx).copy_from_slice(base_score);
        }

        // Pre-allocate feature buffer for the block
        let actual_block_size = block_size.min(num_rows);
        let mut feature_buffer: Vec<Vec<f32>> = (0..actual_block_size)
            .map(|_| vec![f32::NAN; num_features])
            .collect();

        // Process in blocks
        for block_start in (0..num_rows).step_by(block_size) {
            let block_end = (block_start + block_size).min(num_rows);
            let current_block_size = block_end - block_start;

            // Load features for this block
            for (i, row_idx) in (block_start..block_end).enumerate() {
                features.copy_row(row_idx, &mut feature_buffer[i]);
            }

            // Process all trees for this block
            for (tree_idx, (tree, group)) in self.forest.trees_with_groups().enumerate() {
                let weight = weights[tree_idx];
                for i in 0..current_block_size {
                    let row_idx = block_start + i;
                    let leaf_value = visitor.visit_tree(&tree, &feature_buffer[i]);
                    output.row_mut(row_idx)[group as usize] += leaf_value * weight;
                }
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;
    use crate::forest::SoAForest;
    use crate::predict::Predictor;
    use crate::trees::{ScalarLeaf, TreeBuilder};

    fn build_simple_tree(
        left_val: f32,
        right_val: f32,
        threshold: f32,
    ) -> crate::trees::SoATreeStorage<ScalarLeaf> {
        let mut builder = TreeBuilder::new();
        builder.add_split(0, threshold, true, 1, 2);
        builder.add_leaf(ScalarLeaf(left_val));
        builder.add_leaf(ScalarLeaf(right_val));
        builder.build()
    }

    #[test]
    fn block_predictor_matches_regular() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let regular = Predictor::new(&forest);
        let block = BlockPredictor::new(&forest);

        // Test with various batch sizes
        for num_rows in [1, 10, 64, 100, 128, 200] {
            let data: Vec<f32> = (0..num_rows).map(|i| (i as f32) / (num_rows as f32)).collect();
            let features = DenseMatrix::from_vec(data, num_rows, 1);

            let regular_output = regular.predict(&features);
            let block_output = block.predict(&features);

            assert_eq!(regular_output.shape(), block_output.shape());
            for row_idx in 0..num_rows {
                assert_eq!(
                    regular_output.row(row_idx),
                    block_output.row(row_idx),
                    "Mismatch at row {} with {} total rows",
                    row_idx,
                    num_rows
                );
            }
        }
    }

    #[test]
    fn block_predictor_multiclass() {
        let mut forest = SoAForest::new(3);
        forest.push_tree(build_simple_tree(0.1, 0.9, 0.5), 0);
        forest.push_tree(build_simple_tree(0.2, 0.8, 0.5), 1);
        forest.push_tree(build_simple_tree(0.3, 0.7, 0.5), 2);

        let block = BlockPredictor::new(&forest);

        let features = DenseMatrix::from_vec(vec![0.3, 0.7], 2, 1);
        let output = block.predict(&features);

        assert_eq!(output.shape(), (2, 3));
        assert_eq!(output.row(0), &[0.1, 0.2, 0.3]); // all go left
        assert_eq!(output.row(1), &[0.9, 0.8, 0.7]); // all go right
    }

    #[test]
    fn block_predictor_with_base_score() {
        let mut forest = SoAForest::for_regression().with_base_score(vec![0.5]);
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let block = BlockPredictor::new(&forest);

        let features = DenseMatrix::from_vec(vec![0.3], 1, 1);
        let output = block.predict(&features);

        // base_score + leaf = 0.5 + 1.0 = 1.5
        assert_eq!(output.row(0), &[1.5]);
    }

    #[test]
    fn block_predictor_weighted() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let regular = Predictor::new(&forest);
        let block = BlockPredictor::new(&forest);

        let features = DenseMatrix::from_vec(vec![0.3, 0.7], 2, 1);
        let weights = &[1.0, 0.5];

        let regular_output = regular.predict_weighted(&features, weights);
        let block_output = block.predict_weighted(&features, weights);

        assert_eq!(regular_output.shape(), block_output.shape());
        for row_idx in 0..2 {
            let r = regular_output.row(row_idx);
            let b = block_output.row(row_idx);
            assert!(
                (r[0] - b[0]).abs() < 1e-6,
                "Mismatch at row {}: {:?} vs {:?}",
                row_idx,
                r,
                b
            );
        }
    }

    #[test]
    fn block_predictor_custom_block_size() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let config = BlockConfig { block_size: 16 };
        let block = BlockPredictor::with_config(&forest, config);

        assert_eq!(block.block_size(), 16);

        let features = DenseMatrix::from_vec(vec![0.3; 100], 100, 1);
        let output = block.predict(&features);

        assert_eq!(output.shape(), (100, 1));
        for row_idx in 0..100 {
            assert_eq!(output.row(row_idx), &[1.0]);
        }
    }

    #[test]
    fn block_predictor_empty_input() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let block = BlockPredictor::new(&forest);

        let features = DenseMatrix::from_vec(vec![], 0, 1);
        let output = block.predict(&features);

        assert_eq!(output.shape(), (0, 1));
    }

    #[test]
    fn block_predictor_single_row() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let block = BlockPredictor::new(&forest);

        let features = DenseMatrix::from_vec(vec![0.3], 1, 1);
        let output = block.predict(&features);

        assert_eq!(output.row(0), &[1.0]);
    }
}
