//! Nested parameter groups for GBDT configuration.
//!
//! These structs are used by both high-level [`GBDTConfig`] and mid-level
//! trainer parameters. They provide semantic grouping of related settings.
//!
//! # Design Rationale
//!
//! Parameters are grouped by concern:
//! - [`TreeParams`]: Tree structure (max_depth, max_leaves, growth strategy)
//! - [`RegularizationParams`]: L1/L2 regularization and split constraints
//! - [`SamplingParams`]: Row and column subsampling rates
//!
//! Each group has sensible defaults and validation methods.

use crate::training::gbdt::GrowthStrategy;

// =============================================================================
// TreeParams
// =============================================================================

/// Tree structure parameters.
///
/// Controls how individual trees are grown - their maximum size and growth pattern.
///
/// # Example
///
/// ```
/// use boosters::model::gbdt::TreeParams;
/// use boosters::training::GrowthStrategy;
///
/// // Depth-limited trees (XGBoost style)
/// let params = TreeParams::depth_wise(8);
///
/// // Leaf-limited trees (LightGBM style)
/// let params = TreeParams::leaf_wise(63);
/// ```
#[derive(Debug, Clone)]
pub struct TreeParams {
    /// Tree growth strategy (depth-wise or leaf-wise with size limits).
    pub growth_strategy: GrowthStrategy,
    /// Maximum categories for one-hot encoding categorical splits.
    /// Categories beyond this threshold use partition-based splits.
    pub max_onehot_cats: u32,
}

impl Default for TreeParams {
    fn default() -> Self {
        Self {
            growth_strategy: GrowthStrategy::default(),
            max_onehot_cats: 4,
        }
    }
}

impl TreeParams {
    /// Create depth-wise growth with specified max depth.
    pub fn depth_wise(max_depth: u32) -> Self {
        Self {
            growth_strategy: GrowthStrategy::DepthWise { max_depth },
            ..Default::default()
        }
    }

    /// Create leaf-wise growth with specified max leaves.
    pub fn leaf_wise(max_leaves: u32) -> Self {
        Self {
            growth_strategy: GrowthStrategy::LeafWise { max_leaves },
            ..Default::default()
        }
    }

    /// Set max categories for one-hot encoding.
    pub fn with_max_onehot_cats(mut self, max_onehot_cats: u32) -> Self {
        self.max_onehot_cats = max_onehot_cats;
        self
    }

    /// Get the maximum depth (if using depth-wise growth).
    pub fn max_depth(&self) -> Option<u32> {
        match self.growth_strategy {
            GrowthStrategy::DepthWise { max_depth } => Some(max_depth),
            GrowthStrategy::LeafWise { .. } => None,
        }
    }

    /// Get the maximum leaves (if using leaf-wise growth).
    pub fn max_leaves(&self) -> Option<u32> {
        match self.growth_strategy {
            GrowthStrategy::DepthWise { .. } => None,
            GrowthStrategy::LeafWise { max_leaves } => Some(max_leaves),
        }
    }
}

// =============================================================================
// RegularizationParams
// =============================================================================

/// Regularization parameters.
///
/// Controls L1/L2 regularization and split constraints. These prevent overfitting
/// and control tree complexity.
///
/// # Example
///
/// ```
/// use boosters::model::gbdt::RegularizationParams;
///
/// let params = RegularizationParams {
///     lambda: 1.0,      // L2 regularization
///     alpha: 0.1,       // L1 regularization
///     min_child_weight: 5.0,  // Require more samples per leaf
///     min_gain: 0.01,         // Require minimum split gain
///     min_samples_leaf: 1,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct RegularizationParams {
    /// L2 regularization term on leaf weights. Default: 1.0.
    ///
    /// Higher values = more conservative model (smaller leaf weights).
    pub lambda: f32,

    /// L1 regularization term on leaf weights. Default: 0.0.
    ///
    /// Encourages sparse leaf weights (feature selection within leaves).
    pub alpha: f32,

    /// Minimum sum of hessians required in a leaf. Default: 1.0.
    ///
    /// Larger values prevent learning patterns from small subsets.
    pub min_child_weight: f32,

    /// Minimum gain required to make a split. Default: 0.0.
    ///
    /// Higher values = fewer splits, simpler trees.
    pub min_gain: f32,

    /// Minimum number of samples required in a leaf. Default: 1.
    ///
    /// Larger values prevent leaves with very few samples.
    pub min_samples_leaf: u32,
}

impl Default for RegularizationParams {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            alpha: 0.0,
            min_child_weight: 1.0,
            min_gain: 0.0,
            min_samples_leaf: 1,
        }
    }
}

impl RegularizationParams {
    /// Validate parameters.
    ///
    /// # Errors
    ///
    /// Returns error if any parameter is invalid.
    pub fn validate(&self) -> Result<(), ParamValidationError> {
        if self.lambda < 0.0 {
            return Err(ParamValidationError::InvalidLambda(self.lambda));
        }
        if self.alpha < 0.0 {
            return Err(ParamValidationError::InvalidAlpha(self.alpha));
        }
        if self.min_child_weight < 0.0 {
            return Err(ParamValidationError::InvalidMinChildWeight(self.min_child_weight));
        }
        if self.min_gain < 0.0 {
            return Err(ParamValidationError::InvalidMinGain(self.min_gain));
        }
        Ok(())
    }
}

// =============================================================================
// SamplingParams
// =============================================================================

/// Sampling parameters.
///
/// Controls row (sample) and column (feature) subsampling during training.
/// Subsampling provides regularization and can speed up training.
///
/// All rates are in the range (0, 1]. A rate of 1.0 means no sampling.
///
/// # Example
///
/// ```
/// use boosters::model::gbdt::SamplingParams;
///
/// // Subsample 80% of rows and 80% of columns per tree
/// let params = SamplingParams {
///     subsample: 0.8,
///     colsample_bytree: 0.8,
///     colsample_bylevel: 1.0,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Row subsampling ratio per tree. Default: 1.0 (no sampling).
    ///
    /// A value of 0.8 means randomly sample 80% of rows for each tree.
    pub subsample: f32,

    /// Column subsampling ratio per tree. Default: 1.0 (no sampling).
    ///
    /// A value of 0.8 means randomly sample 80% of features for each tree.
    pub colsample_bytree: f32,

    /// Column subsampling ratio per tree level. Default: 1.0 (no sampling).
    ///
    /// Applied multiplicatively with `colsample_bytree`.
    pub colsample_bylevel: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            subsample: 1.0,
            colsample_bytree: 1.0,
            colsample_bylevel: 1.0,
        }
    }
}

impl SamplingParams {
    /// Validate parameters.
    ///
    /// # Errors
    ///
    /// Returns error if any sampling rate is not in (0, 1].
    pub fn validate(&self) -> Result<(), ParamValidationError> {
        if self.subsample <= 0.0 || self.subsample > 1.0 {
            return Err(ParamValidationError::InvalidSubsample(self.subsample));
        }
        if self.colsample_bytree <= 0.0 || self.colsample_bytree > 1.0 {
            return Err(ParamValidationError::InvalidColsampleBytree(self.colsample_bytree));
        }
        if self.colsample_bylevel <= 0.0 || self.colsample_bylevel > 1.0 {
            return Err(ParamValidationError::InvalidColsampleBylevel(self.colsample_bylevel));
        }
        Ok(())
    }

    /// Check if any row sampling is configured.
    pub fn has_row_sampling(&self) -> bool {
        self.subsample < 1.0
    }

    /// Check if any column sampling is configured.
    pub fn has_col_sampling(&self) -> bool {
        self.colsample_bytree < 1.0 || self.colsample_bylevel < 1.0
    }
}

// =============================================================================
// Validation Errors
// =============================================================================

/// Parameter validation error.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ParamValidationError {
    /// Lambda (L2 regularization) must be >= 0.
    #[error("lambda must be >= 0, got {0}")]
    InvalidLambda(f32),

    /// Alpha (L1 regularization) must be >= 0.
    #[error("alpha must be >= 0, got {0}")]
    InvalidAlpha(f32),

    /// min_child_weight must be >= 0.
    #[error("min_child_weight must be >= 0, got {0}")]
    InvalidMinChildWeight(f32),

    /// min_gain must be >= 0.
    #[error("min_gain must be >= 0, got {0}")]
    InvalidMinGain(f32),

    /// subsample must be in (0, 1].
    #[error("subsample must be in (0, 1], got {0}")]
    InvalidSubsample(f32),

    /// colsample_bytree must be in (0, 1].
    #[error("colsample_bytree must be in (0, 1], got {0}")]
    InvalidColsampleBytree(f32),

    /// colsample_bylevel must be in (0, 1].
    #[error("colsample_bylevel must be in (0, 1], got {0}")]
    InvalidColsampleBylevel(f32),

    /// learning_rate must be > 0.
    #[error("learning_rate must be > 0, got {0}")]
    InvalidLearningRate(f32),

    /// n_trees must be > 0.
    #[error("n_trees must be > 0, got {0}")]
    InvalidNTrees(u32),
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_params_default() {
        let params = TreeParams::default();
        assert!(matches!(params.growth_strategy, GrowthStrategy::DepthWise { max_depth: 6 }));
        assert_eq!(params.max_onehot_cats, 4);
    }

    #[test]
    fn tree_params_depth_wise() {
        let params = TreeParams::depth_wise(10);
        assert_eq!(params.max_depth(), Some(10));
        assert_eq!(params.max_leaves(), None);
    }

    #[test]
    fn tree_params_leaf_wise() {
        let params = TreeParams::leaf_wise(63);
        assert_eq!(params.max_depth(), None);
        assert_eq!(params.max_leaves(), Some(63));
    }

    #[test]
    fn regularization_params_default() {
        let params = RegularizationParams::default();
        assert_eq!(params.lambda, 1.0);
        assert_eq!(params.alpha, 0.0);
        assert_eq!(params.min_child_weight, 1.0);
        assert_eq!(params.min_gain, 0.0);
        assert_eq!(params.min_samples_leaf, 1);
    }

    #[test]
    fn regularization_params_validation() {
        let valid = RegularizationParams::default();
        assert!(valid.validate().is_ok());

        let invalid_lambda = RegularizationParams { lambda: -1.0, ..Default::default() };
        assert!(matches!(invalid_lambda.validate(), Err(ParamValidationError::InvalidLambda(_))));

        let invalid_alpha = RegularizationParams { alpha: -0.1, ..Default::default() };
        assert!(matches!(invalid_alpha.validate(), Err(ParamValidationError::InvalidAlpha(_))));
    }

    #[test]
    fn sampling_params_default() {
        let params = SamplingParams::default();
        assert_eq!(params.subsample, 1.0);
        assert_eq!(params.colsample_bytree, 1.0);
        assert_eq!(params.colsample_bylevel, 1.0);
        assert!(!params.has_row_sampling());
        assert!(!params.has_col_sampling());
    }

    #[test]
    fn sampling_params_validation() {
        let valid = SamplingParams::default();
        assert!(valid.validate().is_ok());

        let valid_sampling = SamplingParams { subsample: 0.8, colsample_bytree: 0.9, colsample_bylevel: 1.0 };
        assert!(valid_sampling.validate().is_ok());
        assert!(valid_sampling.has_row_sampling());
        assert!(valid_sampling.has_col_sampling());

        let invalid_zero = SamplingParams { subsample: 0.0, ..Default::default() };
        assert!(matches!(invalid_zero.validate(), Err(ParamValidationError::InvalidSubsample(_))));

        let invalid_over = SamplingParams { colsample_bytree: 1.5, ..Default::default() };
        assert!(matches!(invalid_over.validate(), Err(ParamValidationError::InvalidColsampleBytree(_))));
    }
}
