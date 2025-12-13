//! Gain computation and regularization parameters.

// =============================================================================
// Gain Parameters
// =============================================================================

/// Parameters for split gain computation and leaf weight calculation.
///
/// These parameters are static for the lifetime of training and control
/// regularization and splitting constraints.
#[derive(Clone, Debug)]
pub struct GainParams {
    /// L2 regularization (lambda).
    pub reg_lambda: f32,
    /// L1 regularization (alpha).
    pub reg_alpha: f32,
    /// Minimum split gain (gamma).
    pub min_gain: f32,
    /// Minimum sum of hessians per child.
    pub min_child_weight: f32,
    /// Minimum samples per child.
    pub min_samples_leaf: u32,
}

impl Default for GainParams {
    fn default() -> Self {
        Self {
            reg_lambda: 1.0,
            reg_alpha: 0.0,
            min_gain: 0.0,
            min_child_weight: 1.0,
            min_samples_leaf: 1,
        }
    }
}

impl GainParams {
    /// Compute the split gain using XGBoost formula.
    ///
    /// ```text
    /// gain = 0.5 * [G_L²/(H_L + λ) + G_R²/(H_R + λ) - G_P²/(H_P + λ)] - γ
    /// ```
    ///
    /// Where:
    /// - G_L, G_R, G_P = gradient sums for left, right, parent
    /// - H_L, H_R, H_P = hessian sums for left, right, parent
    /// - λ = L2 regularization (reg_lambda)
    /// - γ = minimum gain threshold (min_gain)
    #[inline]
    pub fn compute_gain(
        &self,
        grad_left: f64,
        hess_left: f64,
        grad_right: f64,
        hess_right: f64,
        grad_parent: f64,
        hess_parent: f64,
    ) -> f32 {
        let lambda = self.reg_lambda as f64;

        let score_left = grad_left * grad_left / (hess_left + lambda);
        let score_right = grad_right * grad_right / (hess_right + lambda);
        let score_parent = grad_parent * grad_parent / (hess_parent + lambda);

        let gain = 0.5 * (score_left + score_right - score_parent) - self.min_gain as f64;

        gain as f32
    }

    /// Check if a split satisfies minimum constraints.
    #[inline]
    pub fn is_valid_split(
        &self,
        hess_left: f64,
        hess_right: f64,
        count_left: u32,
        count_right: u32,
    ) -> bool {
        let min_weight = self.min_child_weight as f64;
        let min_samples = self.min_samples_leaf;

        hess_left >= min_weight
            && hess_right >= min_weight
            && count_left >= min_samples
            && count_right >= min_samples
    }

    /// Compute leaf weight with L1 and L2 regularization.
    ///
    /// ```text
    /// weight = -sign(G) × max(0, |G| - α) / (H + λ)
    /// ```
    ///
    /// Where:
    /// - G = gradient sum
    /// - H = hessian sum
    /// - α = L1 regularization (soft thresholding)
    /// - λ = L2 regularization
    #[inline]
    pub fn compute_leaf_weight(&self, grad_sum: f64, hess_sum: f64) -> f32 {
        let lambda = self.reg_lambda as f64;
        let alpha = self.reg_alpha as f64;

        if alpha == 0.0 {
            // No L1: simple Newton step
            (-grad_sum / (hess_sum + lambda)) as f32
        } else {
            // L1: soft thresholding
            let abs_grad = grad_sum.abs();
            if abs_grad <= alpha {
                0.0
            } else {
                let sign = if grad_sum > 0.0 { -1.0 } else { 1.0 };
                (sign * (abs_grad - alpha) / (hess_sum + lambda)) as f32
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gain_computation() {
        let params = GainParams::default();

        // Simple case: symmetric split
        let gain = params.compute_gain(
            10.0, 5.0,  // left: G=10, H=5
            -10.0, 5.0, // right: G=-10, H=5
            0.0, 10.0,  // parent: G=0, H=10
        );

        // score_left = 100/6 ≈ 16.67, score_right = 100/6 ≈ 16.67, score_parent = 0/11 = 0
        // gain = 0.5 * (16.67 + 16.67 - 0) ≈ 16.67
        assert!(gain > 0.0);
        assert!((gain - 16.666).abs() < 0.01);
    }

    #[test]
    fn test_valid_split_check() {
        let params = GainParams {
            min_child_weight: 5.0,
            min_samples_leaf: 10,
            ..Default::default()
        };

        assert!(params.is_valid_split(5.0, 5.0, 10, 10));
        assert!(!params.is_valid_split(4.0, 5.0, 10, 10)); // left hess too small
        assert!(!params.is_valid_split(5.0, 5.0, 9, 10)); // left count too small
    }

    #[test]
    fn test_leaf_weight_no_l1() {
        let params = GainParams::default();
        let weight = params.compute_leaf_weight(-10.0, 5.0);
        // weight = -(-10) / (5 + 1) = 10/6 ≈ 1.67
        assert!((weight - 1.666).abs() < 0.01);
    }

    #[test]
    fn test_leaf_weight_with_l1() {
        let params = GainParams {
            reg_alpha: 2.0, // L1 threshold
            ..Default::default()
        };

        // Gradient below threshold
        let weight1 = params.compute_leaf_weight(-1.0, 5.0);
        assert_eq!(weight1, 0.0); // Shrunk to zero

        // Gradient above threshold
        let weight2 = params.compute_leaf_weight(-10.0, 5.0);
        // weight = -(abs(-10) - 2) / (5 + 1) = -8/6 ≈ 1.33
        assert!((weight2 - 1.333).abs() < 0.01);
    }
}
