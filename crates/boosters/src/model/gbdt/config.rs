//! High-level GBDT configuration with builder pattern.
//!
//! [`GBDTConfig`] provides a unified configuration for GBDT model training.
//! The builder pattern (via `bon`) provides a fluent API with validation at build time.
//!
//! # Example
//!
//! ```
//! use boosters::model::gbdt::GBDTConfig;
//! use boosters::training::{Objective, Metric};
//!
//! // All defaults
//! let config = GBDTConfig::builder().build().unwrap();
//!
//! // Customize objective and hyperparameters
//! use boosters::training::GrowthStrategy;
//! let config = GBDTConfig::builder()
//!     .objective(Objective::LogisticLoss)
//!     .n_trees(200)
//!     .learning_rate(0.1)
//!     .growth_strategy(GrowthStrategy::DepthWise { max_depth: 8 })
//!     .subsample(0.8)
//!     .build()
//!     .unwrap();
//! ```

use crate::data::BinningConfig;
use crate::training::Verbosity;
use crate::training::gbdt::{GrowthStrategy, LinearLeafConfig};
use crate::training::{Metric, Objective};
use bon::Builder;

// =============================================================================
// ConfigError
// =============================================================================

/// Errors that can occur during configuration validation.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    /// Learning rate must be positive.
    InvalidLearningRate(f32),
    /// Number of trees must be at least 1.
    InvalidNTrees,
    /// Invalid sampling ratio (must be in (0, 1]).
    InvalidSamplingRatio { field: &'static str, value: f32 },
    /// Invalid regularization parameter.
    InvalidRegularization { field: &'static str, value: f32 },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLearningRate(v) => {
                write!(f, "learning_rate must be positive, got {}", v)
            }
            Self::InvalidNTrees => write!(f, "n_trees must be at least 1"),
            Self::InvalidSamplingRatio { field, value } => {
                write!(f, "{} must be in (0, 1], got {}", field, value)
            }
            Self::InvalidRegularization { field, value } => {
                write!(f, "{} must be non-negative, got {}", field, value)
            }
        }
    }
}

impl std::error::Error for ConfigError {}

// =============================================================================
// GBDTConfig
// =============================================================================

/// High-level configuration for GBDT model training.
///
/// Uses flat parameter structure for simplicity. The builder pattern
/// (via `bon`) provides a fluent API with validation at build time.
///
/// # Structure
///
/// - **Objective & Metric**: What to optimize and how to measure progress
/// - **Boosting**: Core parameters like `n_trees` and `learning_rate`
/// - **Tree**: Tree structure (`growth_strategy`, `max_onehot_cats`)
/// - **Regularization**: Overfitting control (`lambda`, `alpha`, `min_child_weight`, etc.)
/// - **Sampling**: Data subsampling (`subsample`, `colsample_bytree`, `colsample_bylevel`)
/// - **Early Stopping**: Automatic training termination
/// - **Resources**: Threading and caching
///
/// # Example
///
/// ```
/// use boosters::model::gbdt::GBDTConfig;
///
/// // Default config: regression with squared loss
/// let config = GBDTConfig::builder().build().unwrap();
///
/// // Classification with early stopping
/// use boosters::training::{Objective, Metric};
/// let config = GBDTConfig::builder()
///     .objective(Objective::LogisticLoss)
///     .metric(Metric::Auc)
///     .n_trees(500)
///     .early_stopping_rounds(10)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, Builder)]
#[builder(
    derive(Debug),
    finish_fn(vis = "", name = __build_internal)
)]
pub struct GBDTConfig {
    // === Objective & Metric ===
    /// Loss function for training. Default: `SquaredLoss` (regression).
    #[builder(default)]
    pub objective: Objective,

    /// Evaluation metric.
    ///
    /// If not set explicitly, a default is derived from [`Objective`].
    ///
    /// To disable evaluation entirely, set [`Metric::None`].
    #[builder(default = crate::training::default_metric_for_objective(&objective))]
    pub metric: Metric,

    // === Boosting parameters ===
    /// Number of boosting rounds (trees to train). Default: 100.
    #[builder(default = 100)]
    pub n_trees: u32,

    /// Learning rate (shrinkage). Default: 0.3.
    ///
    /// Smaller values require more trees but often produce better models.
    /// Typical values: 0.01 - 0.3.
    #[builder(default = 0.3)]
    pub learning_rate: f32,

    // === Tree parameters ===
    /// Tree growth strategy (depth-wise or leaf-wise with size limits).
    #[builder(default)]
    pub growth_strategy: GrowthStrategy,

    /// Maximum categories for one-hot encoding categorical splits.
    /// Categories beyond this threshold use partition-based splits.
    #[builder(default = 4)]
    pub max_onehot_cats: u32,

    // === Regularization parameters ===
    /// L2 regularization term on leaf weights. Default: 1.0.
    ///
    /// Higher values = more conservative model (smaller leaf weights).
    #[builder(default = 1.0)]
    pub lambda: f32,

    /// L1 regularization term on leaf weights. Default: 0.0.
    ///
    /// Encourages sparse leaf weights (feature selection within leaves).
    #[builder(default = 0.0)]
    pub alpha: f32,

    /// Minimum sum of hessians required in a leaf. Default: 1.0.
    ///
    /// Larger values prevent learning patterns from small subsets.
    #[builder(default = 1.0)]
    pub min_child_weight: f32,

    /// Minimum gain required to make a split. Default: 0.0.
    ///
    /// Higher values = fewer splits, simpler trees.
    #[builder(default = 0.0)]
    pub min_gain: f32,

    /// Minimum number of samples required in a leaf. Default: 1.
    ///
    /// Larger values prevent leaves with very few samples.
    #[builder(default = 1)]
    pub min_samples_leaf: u32,

    // === Sampling parameters ===
    /// Row subsampling ratio per tree. Default: 1.0 (no sampling).
    ///
    /// A value of 0.8 means randomly sample 80% of rows for each tree.
    #[builder(default = 1.0)]
    pub subsample: f32,

    /// Column subsampling ratio per tree. Default: 1.0 (no sampling).
    ///
    /// A value of 0.8 means randomly sample 80% of features for each tree.
    #[builder(default = 1.0)]
    pub colsample_bytree: f32,

    /// Column subsampling ratio per tree level. Default: 1.0 (no sampling).
    ///
    /// Applied multiplicatively with `colsample_bytree`.
    #[builder(default = 1.0)]
    pub colsample_bylevel: f32,

    // === Binning ===
    /// Binning configuration for quantizing continuous features.
    /// Default: 256 bins with quantile binning, bundling enabled.
    #[builder(default)]
    pub binning: BinningConfig,

    // === Linear leaves ===
    /// Linear leaf configuration. If set, fit linear models in leaves.
    pub linear_leaves: Option<LinearLeafConfig>,

    // === Early stopping ===
    /// Stop training if no improvement for this many rounds.
    /// `None` disables early stopping.
    pub early_stopping_rounds: Option<u32>,

    // === Resource control ===
    /// Histogram cache size (number of slots). Default: 8.
    #[builder(default = 8)]
    pub cache_size: usize,

    // === Reproducibility ===
    /// Random seed. Default: 42.
    #[builder(default = 42)]
    pub seed: u64,

    // === Logging ===
    /// Verbosity level. Default: `Silent`.
    #[builder(default)]
    pub verbosity: Verbosity,
}

/// Custom finishing function that validates the config.
impl<S: g_b_d_t_config_builder::IsComplete> GBDTConfigBuilder<S> {
    /// Build and validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if any parameter is invalid:
    /// - `learning_rate <= 0`
    /// - `n_trees == 0`
    /// - Sampling ratios outside (0, 1]
    /// - Negative regularization parameters
    pub fn build(self) -> Result<GBDTConfig, ConfigError> {
        let config = self.__build_internal();
        config.validate()?;
        Ok(config)
    }
}

impl GBDTConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if any parameter is invalid.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Learning rate must be positive
        if self.learning_rate <= 0.0 {
            return Err(ConfigError::InvalidLearningRate(self.learning_rate));
        }

        // n_trees must be at least 1
        if self.n_trees == 0 {
            return Err(ConfigError::InvalidNTrees);
        }

        // Validate sampling parameters
        if self.subsample <= 0.0 || self.subsample > 1.0 {
            return Err(ConfigError::InvalidSamplingRatio {
                field: "subsample",
                value: self.subsample,
            });
        }
        if self.colsample_bytree <= 0.0 || self.colsample_bytree > 1.0 {
            return Err(ConfigError::InvalidSamplingRatio {
                field: "colsample_bytree",
                value: self.colsample_bytree,
            });
        }
        if self.colsample_bylevel <= 0.0 || self.colsample_bylevel > 1.0 {
            return Err(ConfigError::InvalidSamplingRatio {
                field: "colsample_bylevel",
                value: self.colsample_bylevel,
            });
        }

        // Validate regularization parameters
        if self.lambda < 0.0 {
            return Err(ConfigError::InvalidRegularization {
                field: "lambda",
                value: self.lambda,
            });
        }
        if self.alpha < 0.0 {
            return Err(ConfigError::InvalidRegularization {
                field: "alpha",
                value: self.alpha,
            });
        }
        if self.min_child_weight < 0.0 {
            return Err(ConfigError::InvalidRegularization {
                field: "min_child_weight",
                value: self.min_child_weight,
            });
        }
        if self.min_gain < 0.0 {
            return Err(ConfigError::InvalidRegularization {
                field: "min_gain",
                value: self.min_gain,
            });
        }

        Ok(())
    }
}

impl Default for GBDTConfig {
    fn default() -> Self {
        Self::builder().build().expect("default config is valid")
    }
}

impl GBDTConfig {
    /// Convert to trainer-level parameters.
    ///
    /// This creates the internal `GBDTParams` used by the trainer from
    /// the high-level configuration.
    pub fn to_trainer_params(&self) -> crate::training::gbdt::GBDTParams {
        use crate::training::gbdt::{GBDTParams, GainParams};
        use crate::training::sampling::{ColSamplingParams, RowSamplingParams};

        // Convert to RowSamplingParams
        let row_sampling = if (self.subsample - 1.0).abs() < 1e-6 {
            RowSamplingParams::None
        } else {
            RowSamplingParams::uniform(self.subsample)
        };

        // Convert to ColSamplingParams
        let col_sampling = if (self.colsample_bytree - 1.0).abs() < 1e-6
            && (self.colsample_bylevel - 1.0).abs() < 1e-6
        {
            ColSamplingParams::None
        } else {
            ColSamplingParams::Sample {
                colsample_bytree: self.colsample_bytree,
                colsample_bylevel: self.colsample_bylevel,
                colsample_bynode: 1.0, // Not exposed in high-level config
            }
        };

        // Convert to GainParams
        let gain = GainParams {
            reg_lambda: self.lambda,
            reg_alpha: self.alpha,
            min_gain: self.min_gain,
            min_child_weight: self.min_child_weight,
            min_samples_leaf: self.min_samples_leaf,
        };

        GBDTParams {
            n_trees: self.n_trees,
            learning_rate: self.learning_rate,
            growth_strategy: self.growth_strategy,
            max_onehot_cats: self.max_onehot_cats,
            gain,
            row_sampling,
            col_sampling,
            cache_size: self.cache_size,
            early_stopping_rounds: self.early_stopping_rounds.unwrap_or(0),
            verbosity: self.verbosity,
            seed: self.seed,
            linear_leaves: self.linear_leaves.clone(),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = GBDTConfig::builder().build();
        assert!(config.is_ok());

        let config = config.unwrap();
        assert_eq!(config.n_trees, 100);
        assert!((config.learning_rate - 0.3).abs() < 1e-6);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_invalid_learning_rate_zero() {
        let result = GBDTConfig::builder().learning_rate(0.0).build();
        assert!(matches!(result, Err(ConfigError::InvalidLearningRate(_))));
    }

    #[test]
    fn test_invalid_learning_rate_negative() {
        let result = GBDTConfig::builder().learning_rate(-0.1).build();
        assert!(matches!(result, Err(ConfigError::InvalidLearningRate(_))));
    }

    #[test]
    fn test_valid_learning_rate_boundary() {
        // 1.0 is valid (matches XGBoost behavior)
        let result = GBDTConfig::builder().learning_rate(1.0).build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_learning_rate_greater_than_one_is_valid() {
        // > 1.0 is allowed (unusual but XGBoost permits it)
        let result = GBDTConfig::builder().learning_rate(1.5).build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_n_trees_zero() {
        let result = GBDTConfig::builder().n_trees(0).build();
        assert!(matches!(result, Err(ConfigError::InvalidNTrees)));
    }

    #[test]
    fn test_valid_n_trees_one() {
        let result = GBDTConfig::builder().n_trees(1).build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_subsample_zero() {
        let result = GBDTConfig::builder().subsample(0.0).build();
        assert!(matches!(
            result,
            Err(ConfigError::InvalidSamplingRatio {
                field: "subsample",
                ..
            })
        ));
    }

    #[test]
    fn test_valid_subsample_one() {
        let result = GBDTConfig::builder().subsample(1.0).build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_subsample_above_one() {
        let result = GBDTConfig::builder().subsample(1.5).build();
        assert!(matches!(
            result,
            Err(ConfigError::InvalidSamplingRatio {
                field: "subsample",
                ..
            })
        ));
    }

    #[test]
    fn test_invalid_colsample_bytree() {
        let result = GBDTConfig::builder().colsample_bytree(0.0).build();
        assert!(matches!(
            result,
            Err(ConfigError::InvalidSamplingRatio {
                field: "colsample_bytree",
                ..
            })
        ));
    }

    #[test]
    fn test_invalid_colsample_bylevel() {
        let result = GBDTConfig::builder().colsample_bylevel(1.5).build();
        assert!(matches!(
            result,
            Err(ConfigError::InvalidSamplingRatio {
                field: "colsample_bylevel",
                ..
            })
        ));
    }

    #[test]
    fn test_custom_objective() {
        use crate::training::Objective;

        let config = GBDTConfig::builder()
            .objective(Objective::LogisticLoss)
            .build();
        assert!(config.is_ok());
    }

    #[test]
    fn test_custom_metric() {
        use crate::training::Metric;

        let config = GBDTConfig::builder().metric(Metric::Auc).build();
        assert!(config.is_ok());
    }

    #[test]
    fn test_early_stopping() {
        let config = GBDTConfig::builder().early_stopping_rounds(10).build();
        assert!(config.is_ok());
        assert_eq!(config.unwrap().early_stopping_rounds, Some(10));
    }

    #[test]
    fn test_growth_strategy_customization() {
        let config = GBDTConfig::builder()
            .growth_strategy(GrowthStrategy::DepthWise { max_depth: 10 })
            .build()
            .unwrap();

        if let GrowthStrategy::DepthWise { max_depth } = config.growth_strategy {
            assert_eq!(max_depth, 10);
        } else {
            panic!("Expected DepthWise growth strategy");
        }
    }

    #[test]
    fn test_regularization_customization() {
        let config = GBDTConfig::builder()
            .lambda(2.0)
            .alpha(0.5)
            .build()
            .unwrap();

        assert!((config.lambda - 2.0).abs() < 1e-6);
        assert!((config.alpha - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_config_default_trait() {
        let config = GBDTConfig::default();
        assert_eq!(config.n_trees, 100);
    }

    #[test]
    fn test_to_trainer_params_conversion() {
        use crate::training::sampling::{ColSamplingParams, RowSamplingParams};

        let config = GBDTConfig::builder()
            .n_trees(200)
            .learning_rate(0.1)
            .lambda(2.0)
            .alpha(0.5)
            .min_gain(0.01)
            .min_child_weight(5.0)
            .min_samples_leaf(3)
            .subsample(0.8)
            .colsample_bytree(0.9)
            .colsample_bylevel(0.7)
            .growth_strategy(GrowthStrategy::DepthWise { max_depth: 10 })
            .seed(123)
            .early_stopping_rounds(10)
            .build()
            .unwrap();

        let params = config.to_trainer_params();

        // Check boosting params
        assert_eq!(params.n_trees, 200);
        assert!((params.learning_rate - 0.1).abs() < 1e-6);
        assert_eq!(params.seed, 123);
        assert_eq!(params.early_stopping_rounds, 10);

        // Check regularization/gain params
        assert!((params.gain.reg_lambda - 2.0).abs() < 1e-6);
        assert!((params.gain.reg_alpha - 0.5).abs() < 1e-6);
        assert!((params.gain.min_gain - 0.01).abs() < 1e-6);
        assert!((params.gain.min_child_weight - 5.0).abs() < 1e-6);
        assert_eq!(params.gain.min_samples_leaf, 3);

        // Check row sampling
        if let RowSamplingParams::Uniform { subsample } = params.row_sampling {
            assert!((subsample - 0.8).abs() < 1e-6);
        } else {
            panic!("Expected Uniform row sampling");
        }

        // Check column sampling
        if let ColSamplingParams::Sample {
            colsample_bytree,
            colsample_bylevel,
            colsample_bynode,
        } = params.col_sampling
        {
            assert!((colsample_bytree - 0.9).abs() < 1e-6);
            assert!((colsample_bylevel - 0.7).abs() < 1e-6);
            assert!((colsample_bynode - 1.0).abs() < 1e-6); // Not exposed in high-level
        } else {
            panic!("Expected Sample column sampling");
        }

        // Check growth strategy
        if let GrowthStrategy::DepthWise { max_depth } = params.growth_strategy {
            assert_eq!(max_depth, 10);
        } else {
            panic!("Expected DepthWise growth strategy");
        }
    }

    #[test]
    fn test_to_trainer_params_no_sampling() {
        use crate::training::sampling::{ColSamplingParams, RowSamplingParams};

        // Default config: subsample=1.0, colsample_*=1.0
        let config = GBDTConfig::builder().build().unwrap();

        let params = config.to_trainer_params();

        // Should be None when all rates are 1.0
        assert!(matches!(params.row_sampling, RowSamplingParams::None));
        assert!(matches!(params.col_sampling, ColSamplingParams::None));
    }
}
