//! Gradient pair for optimization.

/// Gradient and hessian pair for coordinate descent optimization.
///
/// In gradient boosting, we minimize a loss function by computing first (gradient)
/// and second (hessian) derivatives with respect to the predictions.
///
/// - `grad`: First derivative (∂L/∂ŷ) - direction of steepest ascent
/// - `hess`: Second derivative (∂²L/∂ŷ²) - curvature information
///
/// For squared error: grad = (ŷ - y), hess = 1
/// For logistic: grad = (sigmoid(ŷ) - y), hess = sigmoid(ŷ) * (1 - sigmoid(ŷ))
///
/// # Example
///
/// ```
/// use booste_rs::training::GradientPair;
///
/// let gp = GradientPair::new(0.5, 0.25);
/// assert_eq!(gp.grad(), 0.5);
/// assert_eq!(gp.hess(), 0.25);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GradientPair {
    /// First derivative (gradient).
    grad: f32,
    /// Second derivative (hessian).
    hess: f32,
}

impl GradientPair {
    /// Create a new gradient pair.
    #[inline]
    pub fn new(grad: f32, hess: f32) -> Self {
        Self { grad, hess }
    }

    /// Zero gradient pair (neutral element for accumulation).
    pub const ZERO: Self = Self {
        grad: 0.0,
        hess: 0.0,
    };

    /// Get the gradient (first derivative).
    #[inline]
    pub fn grad(&self) -> f32 {
        self.grad
    }

    /// Get the hessian (second derivative).
    #[inline]
    pub fn hess(&self) -> f32 {
        self.hess
    }

    /// Add another gradient pair to this one (in-place).
    #[inline]
    pub fn accumulate(&mut self, other: &GradientPair) {
        self.grad += other.grad;
        self.hess += other.hess;
    }

    /// Scale the gradient pair by a constant.
    #[inline]
    pub fn scale(&mut self, factor: f32) {
        self.grad *= factor;
        self.hess *= factor;
    }

    /// Compute Newton step: -grad / hess (with safety clamp).
    ///
    /// Returns the optimal step size according to Newton's method.
    /// Clamps hessian to avoid division by zero.
    #[inline]
    pub fn newton_step(&self, min_hess: f32) -> f32 {
        let hess_safe = self.hess.max(min_hess);
        -self.grad / hess_safe
    }
}

impl std::ops::Add for GradientPair {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            grad: self.grad + other.grad,
            hess: self.hess + other.hess,
        }
    }
}

impl std::ops::AddAssign for GradientPair {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.accumulate(&other);
    }
}

impl std::ops::Mul<f32> for GradientPair {
    type Output = Self;

    #[inline]
    fn mul(self, factor: f32) -> Self {
        Self {
            grad: self.grad * factor,
            hess: self.hess * factor,
        }
    }
}

impl std::iter::Sum for GradientPair {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(GradientPair::ZERO, |acc, gp| acc + gp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_pair_new() {
        let gp = GradientPair::new(1.0, 0.5);
        assert_eq!(gp.grad(), 1.0);
        assert_eq!(gp.hess(), 0.5);
    }

    #[test]
    fn gradient_pair_zero() {
        let gp = GradientPair::ZERO;
        assert_eq!(gp.grad(), 0.0);
        assert_eq!(gp.hess(), 0.0);
    }

    #[test]
    fn gradient_pair_accumulate() {
        let mut gp = GradientPair::new(1.0, 2.0);
        gp.accumulate(&GradientPair::new(0.5, 0.25));
        assert_eq!(gp.grad(), 1.5);
        assert_eq!(gp.hess(), 2.25);
    }

    #[test]
    fn gradient_pair_scale() {
        let mut gp = GradientPair::new(1.0, 2.0);
        gp.scale(2.0);
        assert_eq!(gp.grad(), 2.0);
        assert_eq!(gp.hess(), 4.0);
    }

    #[test]
    fn gradient_pair_newton_step() {
        // Newton step = -grad / hess = -1.0 / 2.0 = -0.5
        let gp = GradientPair::new(1.0, 2.0);
        assert!((gp.newton_step(1e-6) - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn gradient_pair_newton_step_min_hess() {
        // Small hessian should be clamped
        let gp = GradientPair::new(1.0, 0.0);
        let step = gp.newton_step(1.0);
        assert!((step - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn gradient_pair_add() {
        let gp1 = GradientPair::new(1.0, 2.0);
        let gp2 = GradientPair::new(0.5, 0.25);
        let sum = gp1 + gp2;
        assert_eq!(sum.grad(), 1.5);
        assert_eq!(sum.hess(), 2.25);
    }

    #[test]
    fn gradient_pair_add_assign() {
        let mut gp1 = GradientPair::new(1.0, 2.0);
        gp1 += GradientPair::new(0.5, 0.25);
        assert_eq!(gp1.grad(), 1.5);
        assert_eq!(gp1.hess(), 2.25);
    }

    #[test]
    fn gradient_pair_mul() {
        let gp = GradientPair::new(1.0, 2.0);
        let scaled = gp * 2.0;
        assert_eq!(scaled.grad(), 2.0);
        assert_eq!(scaled.hess(), 4.0);
    }

    #[test]
    fn gradient_pair_sum() {
        let pairs = vec![
            GradientPair::new(1.0, 0.5),
            GradientPair::new(2.0, 1.0),
            GradientPair::new(3.0, 1.5),
        ];
        let total: GradientPair = pairs.into_iter().sum();
        assert_eq!(total.grad(), 6.0);
        assert_eq!(total.hess(), 3.0);
    }
}
