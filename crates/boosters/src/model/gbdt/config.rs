//! High-level GBDT configuration with builder pattern.
//!
//! [`GBDTConfig`] provides a unified configuration for GBDT model training.
//! It composes nested parameter groups for semantic organization and uses
//! the `bon` crate for builder pattern generation with validation.
//!
//! # Example
//!
//! ```
//! use boosters::model::gbdt::{GBDTConfig, TreeParams, SamplingParams};
//! use boosters::training::{Objective, Metric};
//!
//! // All defaults
//! let config = GBDTConfig::builder().build().unwrap();
//!
//! // Customize objective and hyperparameters
//! let config = GBDTConfig::builder()
//!     .objective(Objective::logistic())
//!     .n_trees(200)
//!     .learning_rate(0.1)
//!     .tree(TreeParams::depth_wise(8))
//!     .sampling(SamplingParams { subsample: 0.8, ..Default::default() })
//!     .build()
//!     .unwrap();
//! ```

use std::num::NonZeroUsize;

use bon::Builder;

use super::{RegularizationParams, SamplingParams, TreeParams};
use crate::training::gbdt::LinearLeafConfig;
use crate::training::Verbosity;
use crate::training::{Metric, Objective};

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
/// Uses nested parameter groups for semantic organization. The builder pattern
/// (via `bon`) provides a fluent API with validation at build time.
///
/// # Structure
///
/// - **Objective & Metric**: What to optimize and how to measure progress
/// - **Boosting**: Core parameters like `n_trees` and `learning_rate`
/// - **Tree**: Tree structure via [`TreeParams`]
/// - **Regularization**: Overfitting control via [`RegularizationParams`]
/// - **Sampling**: Data subsampling via [`SamplingParams`]
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
///     .objective(Objective::logistic())
///     .metric(Metric::auc())
///     .n_trees(500)
///     .early_stopping_rounds(10)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, Builder)]
#[builder(
    derive(Clone, Debug),
    finish_fn(vis = "", name = __build_internal)
)]
pub struct GBDTConfig {
    // === Objective & Metric ===
    /// Loss function for training. Default: `SquaredLoss` (regression).
    #[builder(default)]
    pub objective: Objective,

    /// Evaluation metric. If `None`, no eval metrics are computed.
    pub metric: Option<Metric>,

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

    // === Nested parameter groups ===
    /// Tree structure parameters.
    #[builder(default)]
    pub tree: TreeParams,

    /// Regularization parameters.
    #[builder(default)]
    pub regularization: RegularizationParams,

    /// Row and column sampling parameters.
    #[builder(default)]
    pub sampling: SamplingParams,

    // === Linear leaves ===
    /// Linear leaf configuration. If set, fit linear models in leaves.
    pub linear_leaves: Option<LinearLeafConfig>,

    // === Early stopping ===
    /// Stop training if no improvement for this many rounds.
    /// `None` disables early stopping.
    pub early_stopping_rounds: Option<u32>,

    // === Resource control ===
    /// Number of threads. `None` uses all available cores.
    pub n_threads: Option<NonZeroUsize>,

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
    fn validate(&self) -> Result<(), ConfigError> {
        // Learning rate must be positive
        if self.learning_rate <= 0.0 {
            return Err(ConfigError::InvalidLearningRate(self.learning_rate));
        }

        // n_trees must be at least 1
        if self.n_trees == 0 {
            return Err(ConfigError::InvalidNTrees);
        }

        // Validate nested param groups
        self.sampling
            .validate()
            .map_err(|e| Self::convert_param_error(e))?;

        self.regularization
            .validate()
            .map_err(|e| Self::convert_param_error(e))?;

        Ok(())
    }

    /// Convert ParamValidationError to ConfigError.
    fn convert_param_error(e: super::ParamValidationError) -> ConfigError {
        use super::ParamValidationError;
        match e {
            ParamValidationError::InvalidLambda(v) => ConfigError::InvalidRegularization {
                field: "lambda",
                value: v,
            },
            ParamValidationError::InvalidAlpha(v) => ConfigError::InvalidRegularization {
                field: "alpha",
                value: v,
            },
            ParamValidationError::InvalidMinChildWeight(v) => ConfigError::InvalidRegularization {
                field: "min_child_weight",
                value: v,
            },
            ParamValidationError::InvalidMinGain(v) => ConfigError::InvalidRegularization {
                field: "min_gain",
                value: v,
            },
            ParamValidationError::InvalidSubsample(v) => ConfigError::InvalidSamplingRatio {
                field: "subsample",
                value: v,
            },
            ParamValidationError::InvalidColsampleBytree(v) => ConfigError::InvalidSamplingRatio {
                field: "colsample_bytree",
                value: v,
            },
            ParamValidationError::InvalidColsampleBylevel(v) => ConfigError::InvalidSamplingRatio {
                field: "colsample_bylevel",
                value: v,
            },
            ParamValidationError::InvalidLearningRate(v) => ConfigError::InvalidLearningRate(v),
            ParamValidationError::InvalidNTrees(_) => ConfigError::InvalidNTrees,
        }
    }
}

impl Default for GBDTConfig {
    fn default() -> Self {
        Self::builder().build().expect("default config is valid")
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
        let result = GBDTConfig::builder()
            .sampling(SamplingParams {
                subsample: 0.0,
                ..Default::default()
            })
            .build();
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
        let result = GBDTConfig::builder()
            .sampling(SamplingParams {
                subsample: 1.0,
                ..Default::default()
            })
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_subsample_above_one() {
        let result = GBDTConfig::builder()
            .sampling(SamplingParams {
                subsample: 1.5,
                ..Default::default()
            })
            .build();
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
        let result = GBDTConfig::builder()
            .sampling(SamplingParams {
                colsample_bytree: 0.0,
                ..Default::default()
            })
            .build();
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
        let result = GBDTConfig::builder()
            .sampling(SamplingParams {
                colsample_bylevel: 1.5,
                ..Default::default()
            })
            .build();
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
            .objective(Objective::logistic())
            .build();
        assert!(config.is_ok());
    }

    #[test]
    fn test_custom_metric() {
        use crate::training::Metric;

        let config = GBDTConfig::builder().metric(Metric::auc()).build();
        assert!(config.is_ok());
    }

    #[test]
    fn test_early_stopping() {
        let config = GBDTConfig::builder().early_stopping_rounds(10).build();
        assert!(config.is_ok());
        assert_eq!(config.unwrap().early_stopping_rounds, Some(10));
    }

    #[test]
    fn test_tree_params_customization() {
        let config = GBDTConfig::builder()
            .tree(TreeParams::depth_wise(10))
            .build()
            .unwrap();

        if let crate::training::gbdt::GrowthStrategy::DepthWise { max_depth } =
            config.tree.growth_strategy
        {
            assert_eq!(max_depth, 10);
        } else {
            panic!("Expected DepthWise growth strategy");
        }
    }

    #[test]
    fn test_regularization_customization() {
        let config = GBDTConfig::builder()
            .regularization(RegularizationParams {
                lambda: 2.0,
                alpha: 0.5,
                ..Default::default()
            })
            .build()
            .unwrap();

        assert!((config.regularization.lambda - 2.0).abs() < 1e-6);
        assert!((config.regularization.alpha - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_threads_customization() {
        let config = GBDTConfig::builder()
            .n_threads(NonZeroUsize::new(4).unwrap())
            .build()
            .unwrap();
        assert_eq!(config.n_threads, Some(NonZeroUsize::new(4).unwrap()));
    }

    #[test]
    fn test_config_default_trait() {
        let config = GBDTConfig::default();
        assert_eq!(config.n_trees, 100);
    }
}
