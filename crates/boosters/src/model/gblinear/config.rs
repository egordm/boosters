//! High-level GBLinear configuration with builder pattern.
//!
//! [`GBLinearConfig`] provides a unified configuration for GBLinear model training.
//! It uses the `bon` crate for builder pattern generation with validation.
//!
//! # Example
//!
//! ```
//! use boosters::model::gblinear::GBLinearConfig;
//! use boosters::training::{Objective, Metric};
//!
//! // All defaults
//! let config = GBLinearConfig::builder().build().unwrap();
//!
//! // Classification with L1/L2 regularization
//! let config = GBLinearConfig::builder()
//!     .objective(Objective::LogisticLoss)
//!     .n_rounds(200)
//!     .learning_rate(0.1)
//!     .alpha(0.1)  // L1
//!     .lambda(2.0) // L2
//!     .build()
//!     .unwrap();
//! ```

use bon::Builder;

use crate::training::Verbosity;
use crate::training::gblinear::FeatureSelectorKind;
use crate::training::gblinear::UpdateStrategy;
use crate::training::{Metric, Objective};

// =============================================================================
// ConfigError
// =============================================================================

/// Errors that can occur during configuration validation.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    /// Learning rate must be positive.
    InvalidLearningRate(f32),
    /// Number of rounds must be at least 1.
    InvalidNRounds,
    /// Invalid regularization parameter (must be >= 0).
    InvalidRegularization { field: &'static str, value: f32 },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLearningRate(v) => {
                write!(f, "learning_rate must be positive, got {}", v)
            }
            Self::InvalidNRounds => write!(f, "n_rounds must be at least 1"),
            Self::InvalidRegularization { field, value } => {
                write!(f, "{} must be non-negative, got {}", field, value)
            }
        }
    }
}

impl std::error::Error for ConfigError {}

// =============================================================================
// RegularizationParams
// =============================================================================

/// Regularization parameters for GBLinear.
///
/// GBLinear uses only L1 (alpha) and L2 (lambda) regularization.
/// No tree-specific parameters like min_child_weight.
#[derive(Debug, Clone)]
pub struct RegularizationParams {
    /// L1 regularization (alpha). Encourages sparse weights. Default: 0.0.
    pub alpha: f32,
    /// L2 regularization (lambda). Prevents large weights. Default: 0.0.
    pub lambda: f32,
}

impl Default for RegularizationParams {
    fn default() -> Self {
        Self {
            alpha: 0.0,
            lambda: 0.0,
        }
    }
}

impl RegularizationParams {
    /// Validate regularization parameters.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.alpha < 0.0 {
            return Err(ConfigError::InvalidRegularization {
                field: "alpha",
                value: self.alpha,
            });
        }
        if self.lambda < 0.0 {
            return Err(ConfigError::InvalidRegularization {
                field: "lambda",
                value: self.lambda,
            });
        }
        Ok(())
    }
}

// =============================================================================
// GBLinearConfig
// =============================================================================

/// High-level configuration for GBLinear model training.
///
/// Uses the builder pattern (via `bon`) with validation at build time.
///
/// # Structure
///
/// - **Objective & Metric**: What to optimize and how to measure progress
/// - **Boosting**: Core parameters like `n_rounds` and `learning_rate`
/// - **Regularization**: L1 (alpha) and L2 (lambda) penalties
/// - **Update Strategy**: Parallel vs sequential, feature selection
/// - **Early Stopping**: Automatic training termination
/// - **Resources**: Threading
///
/// # Example
///
/// ```
/// use boosters::model::gblinear::GBLinearConfig;
///
/// // Default config: regression with squared loss
/// let config = GBLinearConfig::builder().build().unwrap();
///
/// // Classification with early stopping
/// use boosters::training::{Objective, Metric};
/// let config = GBLinearConfig::builder()
///     .objective(Objective::LogisticLoss)
///     .metric(Metric::Auc)
///     .n_rounds(500)
///     .early_stopping_rounds(10)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, Builder)]
#[builder(
    derive(Debug),
    finish_fn(vis = "", name = __build_internal)
)]
pub struct GBLinearConfig {
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
    /// Number of boosting rounds. Default: 100.
    #[builder(default = 100)]
    pub n_rounds: u32,

    /// Learning rate (eta). Default: 0.5.
    ///
    /// Controls step size for weight updates. Higher values mean faster
    /// convergence but risk overshooting.
    #[builder(default = 0.5)]
    pub learning_rate: f32,

    // === Regularization ===
    /// L1 regularization (alpha). Encourages sparse weights. Default: 0.0.
    #[builder(default = 0.0)]
    pub alpha: f32,

    /// L2 regularization (lambda). Prevents large weights. Default: 1.0.
    ///
    /// Note: Matches XGBoost's `reg_lambda` default (0.0).
    #[builder(default = 0.0)]
    pub lambda: f32,

    // === Update strategy ===
    /// Coordinate descent update strategy. Default: `Shotgun`.
    #[builder(default)]
    pub update_strategy: UpdateStrategy,

    /// Feature selection strategy for coordinate descent. Default: Cyclic.
    #[builder(default)]
    pub feature_selector: FeatureSelectorKind,

    /// Maximum per-coordinate Newton step (stability), in absolute value.
    ///
    /// Set to `0.0` to disable.
    #[builder(default = 0.0)]
    pub max_delta_step: f32,

    // === Early stopping ===
    /// Stop training if no improvement for this many rounds.
    /// `None` disables early stopping.
    pub early_stopping_rounds: Option<u32>,

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
impl<S: g_b_linear_config_builder::IsComplete> GBLinearConfigBuilder<S> {
    /// Build and validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if any parameter is invalid:
    /// - `learning_rate <= 0`
    /// - `n_rounds == 0`
    /// - `alpha < 0` or `lambda < 0`
    pub fn build(self) -> Result<GBLinearConfig, ConfigError> {
        let config = self.__build_internal();
        config.validate()?;
        Ok(config)
    }
}

impl GBLinearConfig {
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

        // n_rounds must be at least 1
        if self.n_rounds == 0 {
            return Err(ConfigError::InvalidNRounds);
        }

        // Validate regularization
        if self.alpha < 0.0 {
            return Err(ConfigError::InvalidRegularization {
                field: "alpha",
                value: self.alpha,
            });
        }
        if self.lambda < 0.0 {
            return Err(ConfigError::InvalidRegularization {
                field: "lambda",
                value: self.lambda,
            });
        }

        Ok(())
    }

    /// Get regularization params as a struct.
    pub fn regularization(&self) -> RegularizationParams {
        RegularizationParams {
            alpha: self.alpha,
            lambda: self.lambda,
        }
    }

    /// Convert to trainer-level parameters.
    ///
    /// This creates the internal `GBLinearParams` used by the trainer from
    /// the high-level configuration.
    pub fn to_trainer_params(&self) -> crate::training::gblinear::GBLinearParams {
        use crate::training::gblinear::GBLinearParams;

        GBLinearParams {
            n_rounds: self.n_rounds,
            learning_rate: self.learning_rate,
            alpha: self.alpha,
            lambda: self.lambda,
            update_strategy: self.update_strategy,
            feature_selector: self.feature_selector,
            seed: self.seed,
            max_delta_step: self.max_delta_step,
            early_stopping_rounds: self.early_stopping_rounds.unwrap_or(0),
            verbosity: self.verbosity,
        }
    }
}

impl Default for GBLinearConfig {
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
        let config = GBLinearConfig::builder().build();
        assert!(config.is_ok());

        let config = config.unwrap();
        assert_eq!(config.n_rounds, 100);
        assert!((config.learning_rate - 0.5).abs() < 1e-6);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_invalid_learning_rate_zero() {
        let result = GBLinearConfig::builder().learning_rate(0.0).build();
        assert!(matches!(result, Err(ConfigError::InvalidLearningRate(_))));
    }

    #[test]
    fn test_invalid_learning_rate_negative() {
        let result = GBLinearConfig::builder().learning_rate(-0.1).build();
        assert!(matches!(result, Err(ConfigError::InvalidLearningRate(_))));
    }

    #[test]
    fn test_invalid_n_rounds_zero() {
        let result = GBLinearConfig::builder().n_rounds(0).build();
        assert!(matches!(result, Err(ConfigError::InvalidNRounds)));
    }

    #[test]
    fn test_valid_n_rounds_one() {
        let result = GBLinearConfig::builder().n_rounds(1).build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_alpha_negative() {
        let result = GBLinearConfig::builder().alpha(-0.1).build();
        assert!(matches!(
            result,
            Err(ConfigError::InvalidRegularization { field: "alpha", .. })
        ));
    }

    #[test]
    fn test_valid_alpha_zero() {
        let result = GBLinearConfig::builder().alpha(0.0).build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_lambda_negative() {
        let result = GBLinearConfig::builder().lambda(-0.1).build();
        assert!(matches!(
            result,
            Err(ConfigError::InvalidRegularization {
                field: "lambda",
                ..
            })
        ));
    }

    #[test]
    fn test_valid_lambda_zero() {
        let result = GBLinearConfig::builder().lambda(0.0).build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_custom_objective() {
        use crate::training::Objective;

        let config = GBLinearConfig::builder()
            .objective(Objective::LogisticLoss)
            .build();
        assert!(config.is_ok());
    }

    #[test]
    fn test_custom_metric() {
        use crate::training::Metric;

        let config = GBLinearConfig::builder().metric(Metric::Auc).build();
        assert!(config.is_ok());
    }

    #[test]
    fn test_early_stopping() {
        let config = GBLinearConfig::builder().early_stopping_rounds(10).build();
        assert!(config.is_ok());
        assert_eq!(config.unwrap().early_stopping_rounds, Some(10));
    }

    #[test]
    fn test_regularization_method() {
        let config = GBLinearConfig::builder()
            .alpha(0.5)
            .lambda(2.0)
            .build()
            .unwrap();

        let reg = config.regularization();
        assert!((reg.alpha - 0.5).abs() < 1e-6);
        assert!((reg.lambda - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_default_trait() {
        let config = GBLinearConfig::default();
        assert_eq!(config.n_rounds, 100);
    }

    #[test]
    fn test_to_trainer_params_conversion() {
        let config = GBLinearConfig::builder()
            .n_rounds(200)
            .learning_rate(0.3)
            .alpha(0.5)
            .lambda(2.0)
            .update_strategy(UpdateStrategy::Sequential)
            .seed(123)
            .max_delta_step(0.25)
            .early_stopping_rounds(10)
            .build()
            .unwrap();

        let params = config.to_trainer_params();

        // Check basic params
        assert_eq!(params.n_rounds, 200);
        assert!((params.learning_rate - 0.3).abs() < 1e-6);
        assert!((params.alpha - 0.5).abs() < 1e-6);
        assert!((params.lambda - 2.0).abs() < 1e-6);
        assert_eq!(params.update_strategy, UpdateStrategy::Sequential);
        assert_eq!(params.seed, 123);
        assert!((params.max_delta_step - 0.25).abs() < 1e-6);
        assert_eq!(params.early_stopping_rounds, 10);
    }

    #[test]
    fn test_to_trainer_params_no_early_stopping() {
        let config = GBLinearConfig::builder().build().unwrap();

        let params = config.to_trainer_params();

        // Early stopping should be 0 when not set
        assert_eq!(params.early_stopping_rounds, 0);
    }
}
