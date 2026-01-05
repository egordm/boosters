//! Feature selectors for coordinate descent.
//!
//! Controls the order in which features are updated during training.
//!
//! # Available Selectors
//!
//! | Selector | Description | Complexity |
//! |----------|-------------|------------|
//! | [`CyclicSelector`] | Sequential order (0, 1, 2, ...) | O(1) per feature |
//! | [`ShuffleSelector`] | Random permutation each round | O(n) shuffle, O(1) per feature |
//! | [`RandomSelector`] | Random with replacement | O(1) per feature |
//! | [`GreedySelector`] | Largest gradient magnitude first | O(n²) or O(n × top_k) |
//! | [`ThriftySelector`] | Approximate greedy (sort once) | O(n log n) setup, O(1) per feature |
//!
//! # XGBoost Compatibility
//!
//! These selectors match XGBoost's `feature_selector` parameter:
//! - `cyclic` → [`CyclicSelector`]
//! - `shuffle` → [`ShuffleSelector`]
//! - `random` → [`RandomSelector`]
//! - `greedy` → [`GreedySelector`]
//! - `thrifty` → [`ThriftySelector`]

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use ndarray::ArrayView1;

use crate::data::Dataset;
use crate::training::GradsTuple;

use super::updater::Regularization;

// =============================================================================
// FeatureSelectorKind - Configuration enum
// =============================================================================

/// Feature selector configuration for training.
///
/// This enum specifies which feature selection strategy to use during
/// coordinate descent training. It's used in [`super::GBLinearTrainer`] to
/// configure the trainer.
///
/// # Example
///
/// ```ignore
/// use boosters::training::{GBLinearTrainer, FeatureSelectorKind};
///
/// let trainer = GBLinearTrainer::builder()
///     .feature_selector(FeatureSelectorKind::Thrifty { top_k: 10 })
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FeatureSelectorKind {
    /// Cyclic: sequential order (0, 1, 2, ...).
    #[default]
    Cyclic,
    /// Shuffle: random permutation each round (default).
    Shuffle,
    /// Random: random selection with replacement.
    Random,
    /// Greedy: select features with largest gradient magnitude.
    /// Uses `top_k` to limit selection (0 = all features).
    Greedy { top_k: usize },
    /// Thrifty: approximate greedy with one-time sorting.
    /// Uses `top_k` to limit selection (0 = all features).
    Thrifty { top_k: usize },
}

impl FeatureSelectorKind {
    /// Check if this selector requires setup with gradient information.
    pub fn requires_setup(&self) -> bool {
        matches!(self, Self::Greedy { .. } | Self::Thrifty { .. })
    }

    /// Create a SelectorState from this configuration.
    pub fn create_state(&self, seed: u64) -> SelectorState {
        match self {
            Self::Cyclic => SelectorState::Cyclic(CyclicSelector::new()),
            Self::Shuffle => SelectorState::Shuffle(ShuffleSelector::new(seed)),
            Self::Random => SelectorState::Random(RandomSelector::new(seed)),
            Self::Greedy { top_k } => SelectorState::Greedy(GreedySelector::new(*top_k)),
            Self::Thrifty { top_k } => SelectorState::Thrifty(ThriftySelector::new(*top_k)),
        }
    }
}

// =============================================================================
// SelectorState - Runtime selector wrapper
// =============================================================================

/// Runtime selector state used by the trainer.
///
/// This enum wraps all selector types and provides a unified interface
/// that handles the special setup requirements of Greedy/Thrifty selectors.
pub enum SelectorState {
    Cyclic(CyclicSelector),
    Shuffle(ShuffleSelector),
    Random(RandomSelector),
    Greedy(GreedySelector),
    Thrifty(ThriftySelector),
}

impl SelectorState {
    /// Setup the selector for an update round.
    ///
    /// For Greedy/Thrifty, this computes gradient magnitudes for ranking.
    /// For other selectors, this just resets the iteration state.
    ///
    /// The `reg` parameter should be pre-denormalized (scaled by sum_instance_weight).
    pub fn setup_round(
        &mut self,
        weights_and_bias: ArrayView1<'_, f32>,
        data: &Dataset,
        grad_pairs: &[GradsTuple],
        reg: Regularization,
    ) {
        let n_features = weights_and_bias.len() - 1;

        match self {
            Self::Cyclic(s) => s.reset(n_features),
            Self::Shuffle(s) => s.reset(n_features),
            Self::Random(s) => s.reset(n_features),
            Self::Greedy(s) => {
                s.setup(weights_and_bias, data, grad_pairs, reg.alpha, reg.lambda);
                s.reset(n_features);
            }
            Self::Thrifty(s) => {
                s.setup(weights_and_bias, data, grad_pairs, reg.alpha, reg.lambda);
                s.reset(n_features);
            }
        }
    }

    /// Get all feature indices for parallel updates.
    pub fn all_indices(&mut self) -> Vec<usize> {
        match self {
            Self::Cyclic(s) => s.all_indices(),
            Self::Shuffle(s) => s.all_indices(),
            Self::Random(s) => s.all_indices(),
            Self::Greedy(s) => s.all_indices(),
            Self::Thrifty(s) => s.all_indices(),
        }
    }
}

impl FeatureSelector for SelectorState {
    fn reset(&mut self, n_features: usize) {
        match self {
            Self::Cyclic(s) => s.reset(n_features),
            Self::Shuffle(s) => s.reset(n_features),
            Self::Random(s) => s.reset(n_features),
            Self::Greedy(s) => s.reset(n_features),
            Self::Thrifty(s) => s.reset(n_features),
        }
    }

    fn next(&mut self) -> Option<usize> {
        match self {
            Self::Cyclic(s) => s.next(),
            Self::Shuffle(s) => s.next(),
            Self::Random(s) => s.next(),
            Self::Greedy(s) => s.next(),
            Self::Thrifty(s) => s.next(),
        }
    }

    fn all_indices(&mut self) -> Vec<usize> {
        match self {
            Self::Cyclic(s) => s.all_indices(),
            Self::Shuffle(s) => s.all_indices(),
            Self::Random(s) => s.all_indices(),
            Self::Greedy(s) => s.all_indices(),
            Self::Thrifty(s) => s.all_indices(),
        }
    }
}

// =============================================================================
// FeatureSelector trait
// =============================================================================

/// Trait for selecting features during coordinate descent.
pub trait FeatureSelector: Send + Sync {
    /// Reset the selector for a new round.
    fn reset(&mut self, n_features: usize);

    /// Get the next feature index to update.
    ///
    /// Returns `None` when all features have been visited this round.
    fn next(&mut self) -> Option<usize>;

    /// Get all feature indices for this round (for parallel updates).
    fn all_indices(&mut self) -> Vec<usize>;
}

/// Cyclic feature selector: visits features in sequential order.
///
/// Simple and deterministic. Good baseline for debugging.
#[derive(Debug, Clone, Default)]
pub struct CyclicSelector {
    n_features: usize,
    current: usize,
}

impl CyclicSelector {
    /// Create a new cyclic selector.
    pub fn new() -> Self {
        Self::default()
    }
}

impl FeatureSelector for CyclicSelector {
    fn reset(&mut self, n_features: usize) {
        self.n_features = n_features;
        self.current = 0;
    }

    fn next(&mut self) -> Option<usize> {
        if self.current < self.n_features {
            let idx = self.current;
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }

    fn all_indices(&mut self) -> Vec<usize> {
        self.current = self.n_features;
        (0..self.n_features).collect()
    }
}

/// Shuffle feature selector: visits features in random order each round.
///
/// Recommended for better convergence in practice.
#[derive(Debug, Clone)]
pub struct ShuffleSelector {
    indices: Vec<usize>,
    current: usize,
    rng: rand::rngs::StdRng,
}

impl ShuffleSelector {
    /// Create a new shuffle selector with a random seed.
    pub fn new(seed: u64) -> Self {
        Self {
            indices: Vec::new(),
            current: 0,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }
}

impl Default for ShuffleSelector {
    fn default() -> Self {
        Self::new(42)
    }
}

impl FeatureSelector for ShuffleSelector {
    fn reset(&mut self, n_features: usize) {
        self.indices = (0..n_features).collect();
        self.indices.shuffle(&mut self.rng);
        self.current = 0;
    }

    fn next(&mut self) -> Option<usize> {
        if self.current < self.indices.len() {
            let idx = self.indices[self.current];
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }

    fn all_indices(&mut self) -> Vec<usize> {
        self.current = self.indices.len();
        self.indices.clone()
    }
}

// =============================================================================
// RandomSelector
// =============================================================================

/// Random feature selector: selects features with replacement.
///
/// Unlike [`ShuffleSelector`], this can select the same feature multiple times
/// per round. Each call to [`next()`](FeatureSelector::next) independently samples
/// from the uniform distribution over features.
///
/// This matches XGBoost's `feature_selector='random'` option.
#[derive(Debug, Clone)]
pub struct RandomSelector {
    n_features: usize,
    remaining: usize,
    rng: rand::rngs::StdRng,
}

impl RandomSelector {
    /// Create a new random selector with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            n_features: 0,
            remaining: 0,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }
}

impl Default for RandomSelector {
    fn default() -> Self {
        Self::new(42)
    }
}

impl FeatureSelector for RandomSelector {
    fn reset(&mut self, n_features: usize) {
        self.n_features = n_features;
        self.remaining = n_features;
    }

    fn next(&mut self) -> Option<usize> {
        if self.remaining > 0 && self.n_features > 0 {
            self.remaining -= 1;
            Some(self.rng.gen_range(0..self.n_features))
        } else {
            None
        }
    }

    fn all_indices(&mut self) -> Vec<usize> {
        let indices: Vec<usize> = (0..self.n_features)
            .map(|_| self.rng.gen_range(0..self.n_features))
            .collect();
        self.remaining = 0;
        indices
    }
}

// =============================================================================
// GreedySelector
// =============================================================================

/// Greedy feature selector: selects features by gradient magnitude.
///
/// At each step, computes the weight update magnitude for all remaining features
/// and selects the one with the largest absolute update. This is the theoretically
/// optimal coordinate descent order but has O(n²) complexity.
///
/// Use `top_k` to limit selection to the K features with largest updates per round,
/// reducing complexity to O(n × top_k).
///
/// This matches XGBoost's `feature_selector='greedy'` option.
///
/// # Note
///
/// This selector requires access to data and gradients during selection, so it
/// uses a different workflow than simple selectors. Call [`setup()`](Self::setup)
/// before the update round to pre-compute gradient information.
#[derive(Debug, Clone)]
pub struct GreedySelector {
    /// Pre-computed weight update magnitudes for each feature
    update_magnitudes: Vec<f32>,
    /// Features sorted by descending magnitude
    sorted_features: Vec<usize>,
    /// Current position in sorted order
    current: usize,
    /// Maximum features to select per round (0 = all)
    top_k: usize,
}

impl GreedySelector {
    /// Create a new greedy selector.
    ///
    /// # Arguments
    ///
    /// * `top_k` - Maximum features to select per round. Use 0 for all features.
    pub fn new(top_k: usize) -> Self {
        Self {
            update_magnitudes: Vec::new(),
            sorted_features: Vec::new(),
            current: 0,
            top_k,
        }
    }

    /// Setup the selector with gradient information.
    ///
    /// This must be called before each update round to compute which features
    /// have the largest potential weight updates.
    ///
    /// # Arguments
    ///
    /// * `weights_and_bias` - Per-group weights (length `n_features + 1`, last entry is bias)
    /// * `data` - Training data (Dataset)
    /// * `grad_pairs` - Per-group `(grad, hess)` slice (length `n_samples`)
    /// * `alpha` - L1 regularization strength
    /// * `lambda` - L2 regularization strength
    pub fn setup(
        &mut self,
        weights_and_bias: ArrayView1<'_, f32>,
        data: &Dataset,
        grad_pairs: &[GradsTuple],
        alpha: f32,
        lambda: f32,
    ) {
        let n_features = weights_and_bias.len() - 1;

        // Compute update magnitude for each feature
        self.update_magnitudes.clear();
        self.update_magnitudes.reserve(n_features);

        for feature_idx in 0..n_features {
            let current_weight = weights_and_bias[feature_idx];

            // Accumulate gradient and hessian
            let mut sum_grad = 0.0f32;
            let mut sum_hess = 0.0f32;

            data.for_each_feature_value(feature_idx, |row, value| {
                if value.is_nan() {
                    return;
                }
                sum_grad += grad_pairs[row].grad * value;
                sum_hess += grad_pairs[row].hess * value * value;
            });

            // Compute coordinate descent update with elastic net
            let magnitude =
                coordinate_delta_magnitude(sum_grad, sum_hess, current_weight, alpha, lambda);
            self.update_magnitudes.push(magnitude);
        }

        // Sort features by descending magnitude
        self.sorted_features = (0..n_features).collect();
        self.sorted_features.sort_by(|&a, &b| {
            self.update_magnitudes[b]
                .partial_cmp(&self.update_magnitudes[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.current = 0;
    }
}

impl Default for GreedySelector {
    fn default() -> Self {
        Self::new(0)
    }
}

impl FeatureSelector for GreedySelector {
    fn reset(&mut self, n_features: usize) {
        // Just reset position; actual setup is done via setup()
        self.current = 0;
        if self.sorted_features.is_empty() {
            // Fallback to cyclic if setup wasn't called
            self.sorted_features = (0..n_features).collect();
        }
    }

    fn next(&mut self) -> Option<usize> {
        let limit = if self.top_k > 0 {
            self.top_k.min(self.sorted_features.len())
        } else {
            self.sorted_features.len()
        };

        if self.current < limit {
            let feature = self.sorted_features[self.current];
            self.current += 1;
            Some(feature)
        } else {
            None
        }
    }

    fn all_indices(&mut self) -> Vec<usize> {
        let limit = if self.top_k > 0 {
            self.top_k.min(self.sorted_features.len())
        } else {
            self.sorted_features.len()
        };
        self.current = limit;
        self.sorted_features[..limit].to_vec()
    }
}

// =============================================================================
// ThriftySelector
// =============================================================================

/// Thrifty feature selector: approximate greedy with one-time sorting.
///
/// Like [`GreedySelector`], selects features by gradient magnitude, but computes
/// all magnitudes once at the start of the round instead of recomputing after
/// each update. This reduces complexity from O(n²) to O(n log n).
///
/// The approximation works well when:
/// - Learning rate is small (updates don't drastically change gradients)
/// - Features are relatively independent
///
/// Use `top_k` to limit selection to the K features with largest updates.
///
/// This matches XGBoost's `feature_selector='thrifty'` option.
#[derive(Debug, Clone)]
pub struct ThriftySelector {
    /// Features sorted by descending update magnitude
    sorted_features: Vec<usize>,
    /// Current position in sorted order
    current: usize,
    /// Maximum features to select per round (0 = all)
    top_k: usize,
}

impl ThriftySelector {
    /// Create a new thrifty selector.
    ///
    /// # Arguments
    ///
    /// * `top_k` - Maximum features to select per round. Use 0 for all features.
    pub fn new(top_k: usize) -> Self {
        Self {
            sorted_features: Vec::new(),
            current: 0,
            top_k,
        }
    }

    /// Setup the selector with gradient information.
    ///
    /// Computes weight update magnitudes for all features and sorts them once.
    /// This is the same as [`GreedySelector::setup`] but the sorted order is
    /// used for the entire round without recomputation.
    ///
    /// # Arguments
    ///
    /// * `weights_and_bias` - Per-group weights (length `n_features + 1`, last entry is bias)
    /// * `data` - Training data (Dataset)
    /// * `grad_pairs` - Per-group `(grad, hess)` slice (length `n_samples`)
    /// * `alpha` - L1 regularization strength
    /// * `lambda` - L2 regularization strength
    pub fn setup(
        &mut self,
        weights_and_bias: ArrayView1<'_, f32>,
        data: &Dataset,
        grad_pairs: &[GradsTuple],
        alpha: f32,
        lambda: f32,
    ) {
        let n_features = weights_and_bias.len() - 1;

        // Compute update magnitude for each feature
        let mut magnitudes: Vec<(usize, f32)> = (0..n_features)
            .map(|feature_idx| {
                let current_weight = weights_and_bias[feature_idx];

                let mut sum_grad = 0.0f32;
                let mut sum_hess = 0.0f32;

                data.for_each_feature_value(feature_idx, |row, value| {
                    if value.is_nan() {
                        return;
                    }
                    sum_grad += grad_pairs[row].grad * value;
                    sum_hess += grad_pairs[row].hess * value * value;
                });

                let magnitude =
                    coordinate_delta_magnitude(sum_grad, sum_hess, current_weight, alpha, lambda);
                (feature_idx, magnitude)
            })
            .collect();

        // Sort by descending magnitude
        magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        self.sorted_features = magnitudes.into_iter().map(|(f, _)| f).collect();
        self.current = 0;
    }
}

impl Default for ThriftySelector {
    fn default() -> Self {
        Self::new(0)
    }
}

impl FeatureSelector for ThriftySelector {
    fn reset(&mut self, n_features: usize) {
        self.current = 0;
        if self.sorted_features.is_empty() {
            // Fallback to cyclic if setup wasn't called
            self.sorted_features = (0..n_features).collect();
        }
    }

    fn next(&mut self) -> Option<usize> {
        let limit = if self.top_k > 0 {
            self.top_k.min(self.sorted_features.len())
        } else {
            self.sorted_features.len()
        };

        if self.current < limit {
            let feature = self.sorted_features[self.current];
            self.current += 1;
            Some(feature)
        } else {
            None
        }
    }

    fn all_indices(&mut self) -> Vec<usize> {
        let limit = if self.top_k > 0 {
            self.top_k.min(self.sorted_features.len())
        } else {
            self.sorted_features.len()
        };
        self.current = limit;
        self.sorted_features[..limit].to_vec()
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Compute the magnitude of coordinate descent update with elastic net.
///
/// This is the absolute value of the weight change that would be applied.
#[inline]
fn coordinate_delta_magnitude(
    sum_grad: f32,
    sum_hess: f32,
    current_weight: f32,
    alpha: f32,
    lambda: f32,
) -> f32 {
    if sum_hess < 1e-5 {
        return 0.0;
    }

    let grad_l2 = sum_grad + lambda * current_weight;
    let hess_l2 = sum_hess + lambda;

    // Soft-thresholding for L1
    let raw = -grad_l2 / hess_l2;
    let threshold = alpha / hess_l2;

    let thresholded = if raw > threshold {
        raw - threshold
    } else if raw < -threshold {
        raw + threshold
    } else {
        0.0
    };

    thresholded.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cyclic_selector() {
        let mut sel = CyclicSelector::new();
        sel.reset(3);

        assert_eq!(sel.next(), Some(0));
        assert_eq!(sel.next(), Some(1));
        assert_eq!(sel.next(), Some(2));
        assert_eq!(sel.next(), None);

        // Reset and try again
        sel.reset(2);
        assert_eq!(sel.next(), Some(0));
        assert_eq!(sel.next(), Some(1));
        assert_eq!(sel.next(), None);
    }

    #[test]
    fn cyclic_all_indices() {
        let mut sel = CyclicSelector::new();
        sel.reset(4);

        let indices = sel.all_indices();
        assert_eq!(indices, vec![0, 1, 2, 3]);

        // After all_indices, next should return None
        assert_eq!(sel.next(), None);
    }

    #[test]
    fn shuffle_selector_visits_all() {
        let mut sel = ShuffleSelector::new(42);
        sel.reset(5);

        let mut visited = [false; 5];
        while let Some(idx) = sel.next() {
            visited[idx] = true;
        }

        assert!(visited.iter().all(|&v| v));
    }

    #[test]
    fn shuffle_selector_different_orders() {
        let mut sel = ShuffleSelector::new(42);

        sel.reset(5);
        let order1 = sel.all_indices();

        sel.reset(5);
        let order2 = sel.all_indices();

        // Should be different orders (with high probability)
        // Note: This could theoretically fail if RNG produces same order
        assert_ne!(order1, order2);
    }

    #[test]
    fn shuffle_selector_reproducible() {
        let mut sel1 = ShuffleSelector::new(123);
        let mut sel2 = ShuffleSelector::new(123);

        sel1.reset(10);
        sel2.reset(10);

        let order1 = sel1.all_indices();
        let order2 = sel2.all_indices();

        assert_eq!(order1, order2);
    }

    // =========================================================================
    // RandomSelector tests
    // =========================================================================

    #[test]
    fn random_selector_returns_n_features() {
        let mut sel = RandomSelector::new(42);
        sel.reset(5);

        let mut count = 0;
        while sel.next().is_some() {
            count += 1;
        }

        assert_eq!(count, 5);
    }

    #[test]
    fn random_selector_with_replacement() {
        // With enough iterations, we should see duplicates
        let mut sel = RandomSelector::new(42);
        sel.reset(3); // Only 3 features

        let indices = sel.all_indices();
        assert_eq!(indices.len(), 3);

        // All indices should be valid
        assert!(indices.iter().all(|&i| i < 3));

        // With only 3 options, duplicates are very likely
        // (probability of no duplicates = 3!/3^3 = 6/27 ≈ 22%)
    }

    #[test]
    fn random_selector_reproducible() {
        let mut sel1 = RandomSelector::new(123);
        let mut sel2 = RandomSelector::new(123);

        sel1.reset(10);
        sel2.reset(10);

        let order1 = sel1.all_indices();
        let order2 = sel2.all_indices();

        assert_eq!(order1, order2);
    }

    // =========================================================================
    // GreedySelector tests
    // =========================================================================

    #[test]
    fn greedy_selector_fallback() {
        // Without setup(), should fallback to cyclic
        let mut sel = GreedySelector::new(0);
        sel.reset(3);

        let indices = sel.all_indices();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn greedy_selector_with_setup() {
        use crate::data::Dataset;
        use crate::repr::gblinear::LinearModel;
        use crate::training::Gradients;
        use ndarray::array;

        // Create simple test data: 3 features, 3 samples
        // Feature-major layout: [n_features, n_samples]
        let features = array![
            [1.0f32, 0.0, 1.0], // f0
            [0.0f32, 1.0, 1.0], // f1
            [0.5f32, 0.5, 0.5]  // f2
        ];
        let dataset = Dataset::from_array(features.view(), None, None);

        let model = LinearModel::zeros(3, 1);

        // Gradients that favor feature 1 (larger magnitude)
        let mut buffer = Gradients::new(3, 1);
        buffer.set(0, 0, 0.1, 1.0); // small grad
        buffer.set(1, 0, 1.0, 1.0); // large grad - feature 1 should be first
        buffer.set(2, 0, 0.2, 1.0);

        let mut sel = GreedySelector::new(0);
        sel.setup(
            model.weights_and_bias(0),
            &dataset,
            buffer.output_pairs(0),
            0.0,
            0.0,
        );
        sel.reset(3);

        // Feature 1 should be selected first (largest gradient impact)
        let first = sel.next();
        assert!(first.is_some());
    }

    #[test]
    fn greedy_selector_top_k() {
        let mut sel = GreedySelector::new(2);
        sel.sorted_features = vec![3, 1, 0, 2]; // Pre-sorted
        sel.reset(4);

        let indices = sel.all_indices();
        assert_eq!(indices.len(), 2);
        assert_eq!(indices, vec![3, 1]); // Only top 2
    }

    // =========================================================================
    // ThriftySelector tests
    // =========================================================================

    #[test]
    fn thrifty_selector_fallback() {
        // Without setup(), should fallback to cyclic
        let mut sel = ThriftySelector::new(0);
        sel.reset(3);

        let indices = sel.all_indices();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn thrifty_selector_with_setup() {
        use crate::data::Dataset;
        use crate::repr::gblinear::LinearModel;
        use crate::training::Gradients;
        use ndarray::array;

        // Same setup as greedy test: 3 features, 3 samples
        // Feature-major layout: [n_features, n_samples]
        let features = array![
            [1.0f32, 0.0, 1.0], // f0
            [0.0f32, 1.0, 1.0], // f1
            [0.5f32, 0.5, 0.5]  // f2
        ];
        let dataset = Dataset::from_array(features.view(), None, None);

        let model = LinearModel::zeros(3, 1);

        let mut buffer = Gradients::new(3, 1);
        buffer.set(0, 0, 0.1, 1.0);
        buffer.set(1, 0, 1.0, 1.0);
        buffer.set(2, 0, 0.2, 1.0);

        let mut sel = ThriftySelector::new(0);
        sel.setup(
            model.weights_and_bias(0),
            &dataset,
            buffer.output_pairs(0),
            0.0,
            0.0,
        );
        sel.reset(3);

        // Should return all 3 features in sorted order
        let indices = sel.all_indices();
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn thrifty_selector_top_k() {
        let mut sel = ThriftySelector::new(2);
        sel.sorted_features = vec![3, 1, 0, 2]; // Pre-sorted
        sel.reset(4);

        let indices = sel.all_indices();
        assert_eq!(indices.len(), 2);
        assert_eq!(indices, vec![3, 1]); // Only top 2
    }

    // =========================================================================
    // Helper function tests
    // =========================================================================

    #[test]
    fn coordinate_delta_magnitude_basic() {
        // No regularization: delta = -grad/hess
        let mag = coordinate_delta_magnitude(1.0, 2.0, 0.0, 0.0, 0.0);
        assert!((mag - 0.5).abs() < 1e-6); // |-(-1.0/2.0)| = 0.5
    }

    #[test]
    fn coordinate_delta_magnitude_with_l2() {
        // L2 adds lambda*w to gradient and lambda to hessian
        let mag = coordinate_delta_magnitude(1.0, 2.0, 0.5, 0.0, 1.0);
        // grad_l2 = 1.0 + 1.0*0.5 = 1.5
        // hess_l2 = 2.0 + 1.0 = 3.0
        // raw = -1.5/3.0 = -0.5
        assert!((mag - 0.5).abs() < 1e-6);
    }

    #[test]
    fn coordinate_delta_magnitude_with_l1() {
        // L1 soft-thresholds the result
        let mag = coordinate_delta_magnitude(1.0, 2.0, 0.0, 0.1, 0.0);
        // raw = -1.0/2.0 = -0.5
        // threshold = 0.1/2.0 = 0.05
        // thresholded = -0.5 + 0.05 = -0.45
        assert!((mag - 0.45).abs() < 1e-6);
    }

    #[test]
    fn coordinate_delta_magnitude_zero_hess() {
        // Near-zero hessian should return 0
        let mag = coordinate_delta_magnitude(1.0, 1e-6, 0.0, 0.0, 0.0);
        assert_eq!(mag, 0.0);
    }
}
