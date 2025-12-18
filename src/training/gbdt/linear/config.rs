//! Configuration for linear leaf training.
//!
//! See RFC-0015 for design rationale.

/// Configuration for linear leaf training.
///
/// Controls regularization, convergence, and feature selection.
#[derive(Clone, Debug)]
pub struct LinearLeafConfig {
    /// L2 regularization on coefficients (default: 0.01).
    /// Small default prevents overfitting in small leaves.
    pub lambda: f32,
    /// L1 regularization for sparse coefficients (default: 0.0).
    pub alpha: f32,
    /// Maximum CD iterations per leaf (default: 10).
    pub max_iterations: u32,
    /// Convergence tolerance (default: 1e-6).
    pub tolerance: f64,
    /// Minimum samples in leaf to fit linear model (default: 50).
    pub min_samples: usize,
    /// Coefficient threshold—prune if |coef| < threshold (default: 1e-6).
    pub coefficient_threshold: f32,
    /// Maximum number of features to use in linear model (default: 10).
    /// Limits memory usage and prevents overfitting in deep trees.
    pub max_features: usize,
}

impl Default for LinearLeafConfig {
    fn default() -> Self {
        Self {
            lambda: 0.01,
            alpha: 0.0,
            max_iterations: 10,
            tolerance: 1e-6,
            min_samples: 50,
            coefficient_threshold: 1e-6,
            max_features: 10,
        }
    }
}

impl LinearLeafConfig {
    /// Create a new config with custom minimum samples.
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }

    /// Create a new config with custom regularization.
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }

    /// Create a new config with custom max iterations.
    pub fn with_max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Create a new config with custom max features.
    pub fn with_max_features(mut self, max_features: usize) -> Self {
        self.max_features = max_features;
        self
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
        let config = LinearLeafConfig::default()
            .with_min_samples(100)
            .with_lambda(0.1)
            .with_max_iterations(20);

        assert_eq!(config.min_samples, 100);
        assert_eq!(config.max_iterations, 20);
        assert!((config.lambda - 0.1).abs() < 1e-6);
    }
}
