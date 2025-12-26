//! Configuration for linear leaf training.
//!
//! See RFC-0015 for design rationale.

use bon::Builder;

/// Configuration for linear leaf training.
///
/// Controls regularization, convergence, and feature selection.
#[derive(Clone, Debug, Builder)]
pub struct LinearLeafConfig {
    /// L2 regularization on coefficients (default: 0.01).
    /// Small default prevents overfitting in small leaves.
    #[builder(default = 0.01)]
    pub lambda: f32,
    /// L1 regularization for sparse coefficients (default: 0.0).
    #[builder(default = 0.0)]
    pub alpha: f32,
    /// Maximum CD iterations per leaf (default: 10).
    #[builder(default = 10)]
    pub max_iterations: u32,
    /// Convergence tolerance (default: 1e-6).
    #[builder(default = 1e-6)]
    pub tolerance: f64,
    /// Minimum samples in leaf to fit linear model (default: 50).
    #[builder(default = 50)]
    pub min_samples: usize,
    /// Coefficient threshold—prune if |coef| < threshold (default: 1e-6).
    #[builder(default = 1e-6)]
    pub coefficient_threshold: f32,
    /// Maximum number of features to use in linear model (default: 10).
    /// Limits memory usage and prevents overfitting in deep trees.
    #[builder(default = 10)]
    pub max_features: usize,
}

impl Default for LinearLeafConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LinearLeafConfig::default();
        assert_eq!(config.min_samples, 50);
        assert_eq!(config.max_iterations, 10);
        assert!((config.lambda - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_builder_pattern() {
        let config = LinearLeafConfig::builder()
            .min_samples(100)
            .lambda(0.1)
            .max_iterations(20)
            .build();

        assert_eq!(config.min_samples, 100);
        assert_eq!(config.max_iterations, 20);
        assert!((config.lambda - 0.1).abs() < 1e-6);
    }
}
