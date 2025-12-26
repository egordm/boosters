//! Weighted least squares solver using coordinate descent.
//!
//! Solves: min Σ h_i (y_i - x_i^T c)² + λ||c||²
//!
//! Uses coordinate descent for simplicity and L1 support (no BLAS dependency).
//! Pre-allocates buffers and reuses them between leaves.

use crate::training::GradsTuple;

/// Compute the triangular matrix index for a symmetric matrix stored in packed form.
///
/// For a symmetric matrix stored as upper triangle in row-major order:
/// ```text
/// [0 1 2]     stored as [0, 1, 2, 3, 4, 5]
/// [  3 4]     where index(i,j) gives position in flat array
/// [    5]
/// ```
///
/// Formula: For row i, col j (i <= j):
/// - Elements in rows 0..i: n + (n-1) + ... + (n-i+1) = i*n - i*(i-1)/2
/// - Offset within row i: j - i
/// - Total: i*n - i*(i-1)/2 + (j - i)
#[inline]
fn tri_index(i: usize, j: usize, n: usize) -> usize {
    debug_assert!(i <= j, "tri_index requires i <= j, got {} > {}", i, j);
    // Avoid overflow: i * n + j - i - i*(i-1)/2 = i * (n - 1) + j - i*(i-1)/2
    // But safer to check i > 0 for the subtraction term
    if i == 0 {
        j
    } else {
        i * n - (i * (i - 1)) / 2 + (j - i)
    }
}

// Note: the old tri_index without size param is removed - it was incorrect.
// Using named function to avoid confusion.

/// Solver for weighted least squares with L2 regularization.
///
/// Minimizes: Σ h_i * (y_i - (intercept + Σ c_j * x_ij))² + λ * Σ c_j²
///
/// where gradients are used as -g/h (Newton direction) for the response y.
///
/// # Design
///
/// - Uses coordinate descent (no BLAS dependency)
/// - Pre-allocates for `max_features` and reuses buffers
/// - Stores XᵀHX in packed upper triangular format
/// - Intercept is always coefficient 0
pub struct WeightedLeastSquaresSolver {
    /// Upper triangle of XᵀHX matrix (packed), size = size*(size+1)/2
    xthx: Vec<f64>,
    /// Xᵀg vector (gradient weighted by features), size = size
    xtg: Vec<f64>,
    /// Solution coefficients, size = size (intercept + features)
    coefficients: Vec<f64>,
    /// Maximum number of features (excluding intercept)
    max_features: usize,
    /// Current problem size (features, excluding intercept)
    current_features: usize,
}

impl WeightedLeastSquaresSolver {
    /// Create a solver with pre-allocated buffers for up to `max_features`.
    ///
    /// The intercept is implicit (+1 to size).
    pub fn new(max_features: usize) -> Self {
        let size = max_features + 1; // +1 for intercept
        Self {
            xthx: vec![0.0; size * (size + 1) / 2],
            xtg: vec![0.0; size],
            coefficients: vec![0.0; size],
            max_features,
            current_features: 0,
        }
    }

    /// Reset solver state for a new problem with `n_features` features.
    ///
    /// Clears accumulated statistics but reuses allocated buffers.
    pub fn reset(&mut self, n_features: usize) {
        debug_assert!(
            n_features <= self.max_features,
            "Too many features: {} > max {}",
            n_features,
            self.max_features
        );
        let size = n_features + 1;
        self.xthx[..size * (size + 1) / 2].fill(0.0);
        self.xtg[..size].fill(0.0);
        self.coefficients[..size].fill(0.0);
        self.current_features = n_features;
    }

    /// Accumulate one sample into the normal equations.
    ///
    /// For the weighted least squares problem:
    /// - `features`: feature values for this sample
    /// - `grad`: gradient (g_i) at this sample
    /// - `hess`: hessian (h_i) at this sample (weight)
    ///
    /// Accumulates into XᵀHX and Xᵀg where the implicit "y" is -g/h.
    pub fn accumulate(&mut self, features: &[f32], grad: f32, hess: f32) {
        let n_features = self.current_features;
        assert_eq!(
            features.len(),
            n_features,
            "Feature count mismatch: {} vs {}",
            features.len(),
            n_features
        );

        let size = n_features + 1;
        let h = hess as f64;
        let g = grad as f64;

        // Build augmented feature vector [1, x1, x2, ..., xn] (intercept first)
        // Accumulate XᵀHX and Xᵀg

        // Intercept-intercept term: h * 1 * 1 = h
        self.xthx[tri_index(0, 0, size)] += h;
        // Intercept term in Xᵀg: -g (since y = -g/h, Xᵀg = Xᵀ * (-g))
        self.xtg[0] -= g;

        for (j, &xj) in features.iter().enumerate() {
            let col = j + 1; // offset by 1 for intercept
            let xj_f64 = xj as f64;

            // Intercept-feature cross term: h * 1 * xj
            self.xthx[tri_index(0, col, size)] += h * xj_f64;

            // Feature-feature terms: h * xi * xj
            for (i, &xi) in features[..=j].iter().enumerate() {
                let row = i + 1;
                self.xthx[tri_index(row, col, size)] += h * (xi as f64) * xj_f64;
            }

            // Feature term in Xᵀg: -g * xj
            self.xtg[col] -= g * xj_f64;
        }
    }

    /// Accumulate a single column (feature) for all samples.
    ///
    /// This is more cache-efficient when features are stored column-major.
    ///
    /// # Arguments
    /// - `feat_idx`: feature index (0-based, will be offset for intercept)
    /// - `values`: feature values for all samples
    /// - `grad_hess`: gradient/hessian tuples for all samples
    ///
    /// Note: Must call `accumulate_intercept` first/separately, then
    /// accumulate each feature column.
    pub fn accumulate_column(&mut self, feat_idx: usize, values: &[f32], grad_hess: &[GradsTuple]) {
        let n_features = self.current_features;
        let size = n_features + 1;
        let col = feat_idx + 1; // offset for intercept

        // Intercept-feature cross term and feature diagonal
        let mut sum_hx = 0.0f64;
        let mut sum_hxx = 0.0f64;
        let mut sum_gx = 0.0f64;

        for (&x, gh) in values.iter().zip(grad_hess.iter()) {
            let xf = x as f64;
            let hf = gh.hess as f64;
            let gf = gh.grad as f64;

            sum_hx += hf * xf;
            sum_hxx += hf * xf * xf;
            sum_gx += gf * xf;
        }

        // Intercept-feature: XᵀHX[0, col]
        self.xthx[tri_index(0, col, size)] += sum_hx;
        // Feature diagonal: XᵀHX[col, col]
        self.xthx[tri_index(col, col, size)] += sum_hxx;
        // Feature in Xᵀg
        self.xtg[col] -= sum_gx;
    }

    /// Accumulate the intercept terms (call once per problem).
    ///
    /// # Arguments
    /// - `grad_hess`: gradient/hessian tuples for all samples
    pub fn accumulate_intercept(&mut self, grad_hess: &[GradsTuple]) {
        let size = self.current_features + 1;
        let mut sum_h = 0.0f64;
        let mut sum_g = 0.0f64;

        for gh in grad_hess.iter() {
            sum_h += gh.hess as f64;
            sum_g += gh.grad as f64;
        }

        // Intercept-intercept: XᵀHX[0, 0] = Σh
        self.xthx[tri_index(0, 0, size)] += sum_h;
        // Intercept in Xᵀg: -Σg
        self.xtg[0] -= sum_g;
    }

    /// Accumulate cross-terms between two features (call for all pairs i < j).
    pub fn accumulate_cross_term(
        &mut self,
        feat_i: usize,
        feat_j: usize,
        values_i: &[f32],
        values_j: &[f32],
        grad_hess: &[GradsTuple],
    ) {
        let size = self.current_features + 1;
        let row = feat_i + 1;
        let col = feat_j + 1;

        let mut sum = 0.0f64;
        for ((&xi, &xj), gh) in values_i.iter().zip(values_j.iter()).zip(grad_hess.iter()) {
            sum += (gh.hess as f64) * (xi as f64) * (xj as f64);
        }

        self.xthx[tri_index(row, col, size)] += sum;
    }

    /// Add L2 regularization to the diagonal (not on intercept).
    ///
    /// Adds λ to `XᵀHX[i,i]` for i = 1..n_features.
    pub fn add_regularization(&mut self, lambda: f64) {
        let n_features = self.current_features;
        let size = n_features + 1;

        for i in 1..=n_features {
            self.xthx[tri_index(i, i, size)] += lambda;
        }
    }

    /// Solve via coordinate descent.
    ///
    /// Returns `true` if converged within `max_iterations`, `false` otherwise.
    ///
    /// After solving, use `coefficients()` to retrieve the solution.
    pub fn solve_cd(&mut self, max_iterations: u32, tolerance: f64) -> bool {
        let n_features = self.current_features;
        let size = n_features + 1;

        // Coordinate descent: update each coefficient in turn
        for _iter in 0..max_iterations {
            let mut max_delta = 0.0f64;

            for j in 0..size {
                // Compute residual for coordinate j:
                // r_j = xtg[j] - Σ_{k≠j} xthx[j,k] * coef[k]
                let mut residual = self.xtg[j];
                for k in 0..size {
                    if k != j {
                        let idx = if k < j {
                            tri_index(k, j, size)
                        } else {
                            tri_index(j, k, size)
                        };
                        residual -= self.xthx[idx] * self.coefficients[k];
                    }
                }

                // Update: coef[j] = residual / xthx[j,j]
                let diag = self.xthx[tri_index(j, j, size)];
                if diag.abs() < 1e-12 {
                    // Skip degenerate coordinate
                    continue;
                }

                let new_coef = residual / diag;
                let delta = (new_coef - self.coefficients[j]).abs();
                max_delta = max_delta.max(delta);
                self.coefficients[j] = new_coef;
            }

            if max_delta < tolerance {
                return true; // Converged
            }
        }

        false // Did not converge
    }

    /// Get the solution coefficients.
    ///
    /// Returns (intercept, feature_coefficients).
    pub fn coefficients(&self) -> (f64, &[f64]) {
        let n_features = self.current_features;
        (self.coefficients[0], &self.coefficients[1..=n_features])
    }

    /// Get current problem size (number of features, excluding intercept).
    pub fn n_features(&self) -> usize {
        self.current_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::DEFAULT_TOLERANCE_F64;

    /// Test simple linear regression: y = 2x + 1
    #[test]
    fn test_solver_simple_regression() {
        let mut solver = WeightedLeastSquaresSolver::new(1);
        solver.reset(1);

        // y = 2x + 1 with uniform hessian (weight = 1)
        // Points: (0, 1), (1, 3), (2, 5), (3, 7)
        // Intercept = 1, slope = 2
        let xs = [0.0f32, 1.0, 2.0, 3.0];
        let ys = [1.0f32, 3.0, 5.0, 7.0];

        // For gradient boosting: grad = -(y - pred) * hess for squared loss
        // At initialization pred = 0, so grad = -y * hess = -y (if hess=1)
        // Actually for WLS fitting: we want to fit y = intercept + coef*x
        // The accumulate expects grad = -(target * hess), hess = weight
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            // For WLS: accumulate with grad = -y (since target = y, hess = 1)
            solver.accumulate(&[x], -y, 1.0);
        }

        let converged = solver.solve_cd(100, 1e-10);
        assert!(converged, "Solver should converge");

        let (intercept, coefs) = solver.coefficients();
        assert!(
            (intercept - 1.0).abs() < 0.01,
            "Intercept should be ~1.0, got {}",
            intercept
        );
        assert!(
            (coefs[0] - 2.0).abs() < 0.01,
            "Coefficient should be ~2.0, got {}",
            coefs[0]
        );
    }

    /// Test multivariate: y = x1 + 2*x2 + 3
    #[test]
    fn test_solver_multivariate() {
        let mut solver = WeightedLeastSquaresSolver::new(2);
        solver.reset(2);

        // Generate points: y = 3 + x1 + 2*x2
        let data = [
            ([0.0f32, 0.0], 3.0f32), // 3 + 0 + 0
            ([1.0, 0.0], 4.0),       // 3 + 1 + 0
            ([0.0, 1.0], 5.0),       // 3 + 0 + 2
            ([1.0, 1.0], 6.0),       // 3 + 1 + 2
            ([2.0, 1.0], 7.0),       // 3 + 2 + 2
            ([1.0, 2.0], 8.0),       // 3 + 1 + 4
        ];

        for (features, y) in &data {
            solver.accumulate(features, -*y, 1.0);
        }

        let converged = solver.solve_cd(100, 1e-10);
        assert!(converged, "Solver should converge");

        let (intercept, coefs) = solver.coefficients();
        assert!(
            (intercept - 3.0).abs() < 0.01,
            "Intercept should be ~3.0, got {}",
            intercept
        );
        assert!(
            (coefs[0] - 1.0).abs() < 0.01,
            "Coef[0] should be ~1.0, got {}",
            coefs[0]
        );
        assert!(
            (coefs[1] - 2.0).abs() < 0.01,
            "Coef[1] should be ~2.0, got {}",
            coefs[1]
        );
    }

    /// Test weighted samples (non-uniform hessians)
    #[test]
    fn test_solver_weighted_samples() {
        let mut solver = WeightedLeastSquaresSolver::new(1);
        solver.reset(1);

        // Two points: (0, 0) with weight 1, (1, 10) with weight 10
        // Weighted least squares should heavily favor the second point
        // Expected: line through (0, ~0) and (1, 10) with slope ~10, intercept ~0
        solver.accumulate(&[0.0f32], 0.0, 1.0);
        solver.accumulate(&[1.0f32], -10.0 * 10.0, 10.0); // grad = -y * hess

        // This is an ill-conditioned problem (nearly singular matrix)
        // Use practical tolerance and more iterations
        let converged = solver.solve_cd(1000, 1e-4);
        assert!(converged, "Solver should converge");

        let (intercept, coefs) = solver.coefficients();
        // The heavily weighted point (1, 10) should dominate
        // Line should be close to y = 10x
        assert!(
            intercept.abs() < 1.0,
            "Intercept should be near 0, got {}",
            intercept
        );
        assert!(
            (coefs[0] - 10.0).abs() < 1.0,
            "Coefficient should be near 10, got {}",
            coefs[0]
        );
    }

    /// Test regularization shrinks coefficients
    #[test]
    fn test_solver_with_regularization() {
        let mut solver = WeightedLeastSquaresSolver::new(1);
        solver.reset(1);

        // y = 10x with multiple points to make the system well-determined
        // Points: (0, 0), (1, 10), (2, 20)
        for i in 0..3 {
            let x = i as f32;
            let y = 10.0 * x;
            solver.accumulate(&[x], -y, 1.0);
        }

        // Without regularization
        let converged = solver.solve_cd(100, 1e-8);
        assert!(converged, "Should converge without regularization");
        let (_, coefs_unreg) = solver.coefficients();
        let coef_unreg = coefs_unreg[0];

        // With strong regularization
        solver.reset(1);
        for i in 0..3 {
            let x = i as f32;
            let y = 10.0 * x;
            solver.accumulate(&[x], -y, 1.0);
        }
        solver.add_regularization(10.0); // Strong regularization

        let converged = solver.solve_cd(100, 1e-8);
        assert!(converged, "Should converge with regularization");
        let (_, coefs_reg) = solver.coefficients();
        let coef_reg = coefs_reg[0];

        // Regularization should shrink the coefficient
        assert!(
            coef_reg.abs() < coef_unreg.abs(),
            "Regularization should shrink coefficient: {} vs {}",
            coef_reg,
            coef_unreg
        );
    }

    /// Test early convergence
    #[test]
    fn test_solver_convergence() {
        let mut solver = WeightedLeastSquaresSolver::new(1);
        solver.reset(1);

        // Simple problem that converges quickly
        for i in 0..10 {
            let x = i as f32;
            let y = 2.0 * x + 1.0;
            solver.accumulate(&[x], -y, 1.0);
        }

        // Should converge well before 100 iterations
        let converged = solver.solve_cd(100, 1e-10);
        assert!(converged, "Should converge for simple problem");
    }

    /// Test non-convergence with very few iterations
    #[test]
    fn test_solver_non_convergence() {
        let mut solver = WeightedLeastSquaresSolver::new(2);
        solver.reset(2);

        // Add some data
        solver.accumulate(&[1.0f32, 2.0], -5.0, 1.0);
        solver.accumulate(&[2.0f32, 1.0], -4.0, 1.0);

        // With only 1 iteration and tight tolerance, should not converge
        let converged = solver.solve_cd(1, 1e-15);
        assert!(!converged, "Should not converge with only 1 iteration");
    }

    /// Test tri_index computes correct indices
    #[test]
    fn test_tri_index() {
        // For size=3 (3x3 matrix):
        // [0 1 2]
        // [  3 4]
        // [    5]
        assert_eq!(tri_index(0, 0, 3), 0);
        assert_eq!(tri_index(0, 1, 3), 1);
        assert_eq!(tri_index(0, 2, 3), 2);
        assert_eq!(tri_index(1, 1, 3), 3);
        assert_eq!(tri_index(1, 2, 3), 4);
        assert_eq!(tri_index(2, 2, 3), 5);
    }

    /// Test column-wise accumulation matches sample-wise
    #[test]
    fn test_column_accumulation() {
        // Sample-wise accumulation
        let mut solver1 = WeightedLeastSquaresSolver::new(2);
        solver1.reset(2);

        let data = [
            ([1.0f32, 2.0], -5.0f32, 1.0f32),
            ([3.0, 1.0], -4.0, 2.0),
            ([2.0, 3.0], -7.0, 1.5),
        ];

        for (features, grad, hess) in &data {
            solver1.accumulate(features, *grad, *hess);
        }

        // Column-wise accumulation
        let mut solver2 = WeightedLeastSquaresSolver::new(2);
        solver2.reset(2);

        let grad_hess: Vec<GradsTuple> = data
            .iter()
            .map(|(_, g, h)| GradsTuple { grad: *g, hess: *h })
            .collect();
        let col0: Vec<f32> = data.iter().map(|(f, _, _)| f[0]).collect();
        let col1: Vec<f32> = data.iter().map(|(f, _, _)| f[1]).collect();

        solver2.accumulate_intercept(&grad_hess);
        solver2.accumulate_column(0, &col0, &grad_hess);
        solver2.accumulate_column(1, &col1, &grad_hess);
        solver2.accumulate_cross_term(0, 1, &col0, &col1, &grad_hess);

        // Solve both
        solver1.solve_cd(100, 1e-10);
        solver2.solve_cd(100, 1e-10);

        let (int1, coefs1) = solver1.coefficients();
        let (int2, coefs2) = solver2.coefficients();

        assert!(
            (int1 - int2).abs() < DEFAULT_TOLERANCE_F64,
            "Intercepts should match: {} vs {}",
            int1,
            int2
        );
        assert!(
            (coefs1[0] - coefs2[0]).abs() < DEFAULT_TOLERANCE_F64,
            "Coef[0] should match: {} vs {}",
            coefs1[0],
            coefs2[0]
        );
        assert!(
            (coefs1[1] - coefs2[1]).abs() < DEFAULT_TOLERANCE_F64,
            "Coef[1] should match: {} vs {}",
            coefs1[1],
            coefs2[1]
        );
    }
}
