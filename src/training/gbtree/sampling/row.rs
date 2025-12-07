//! Row sampling strategies for training.
//!
//! Row sampling selects which samples (rows) to use for each boosting round.
//! This provides regularization and can speed up training.
//!
//! # Available Strategies
//!
//! - [`NoSampler`]: No sampling, use all rows (default)
//! - [`RandomSampler`]: Random fraction without replacement
//! - [`GossSampler`]: Gradient-based one-side sampling (LightGBM)
//!
//! # Usage
//!
//! Configure via [`RowSampling`] enum on the trainer:
//!
//! ```ignore
//! use booste_rs::training::gbtree::RowSampling;
//!
//! // No sampling (default)
//! let sampling = RowSampling::None;
//!
//! // Random 80% subsample
//! let sampling = RowSampling::Random { rate: 0.8 };
//!
//! // GOSS with default parameters
//! let sampling = RowSampling::goss_default();
//!
//! // GOSS with custom parameters
//! let sampling = RowSampling::Goss { top_rate: 0.3, other_rate: 0.15 };
//! ```

use std::fmt;

use rand::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::training::GradientBuffer;

// ============================================================================
// RowSampler trait
// ============================================================================

/// Trait for row sampling strategies.
///
/// Implementors produce a [`RowSample`] containing selected row indices
/// and optional per-row weights.
pub trait RowSampler {
    /// Sample rows for single-output models.
    ///
    /// # Arguments
    /// - `num_rows`: Total number of rows to sample from
    /// - `gradients`: Optional gradient values (required for gradient-based methods)
    /// - `seed`: Random seed for reproducibility
    fn sample(&self, num_rows: usize, gradients: Option<&[f32]>, seed: u64) -> RowSample;

    /// Sample rows for multi-output models.
    ///
    /// Each implementor decides how to handle multiple outputs.
    /// - Gradient-agnostic methods ignore the gradients
    /// - Gradient-based methods compute a scalar importance per row
    fn sample_multioutput(&self, grads: &GradientBuffer, seed: u64) -> RowSample;

    /// Returns true if this sampler actually filters rows.
    fn is_enabled(&self) -> bool;
}

// ============================================================================
// RowSample (sampling result)
// ============================================================================

/// Result of row sampling containing selected rows and optional weights.
#[derive(Debug, Clone)]
pub struct RowSample {
    /// Indices of selected rows (sorted for cache efficiency).
    pub indices: Vec<u32>,
    /// Optional weights for each selected row.
    /// - `None`: All rows have weight 1.0 (random sampling or no sampling)
    /// - `Some`: Per-row weights (GOSS - top rows get 1.0, sampled get amplified)
    pub weights: Option<Vec<f32>>,
}

impl RowSample {
    /// Create a sample that includes all rows with uniform weight.
    pub fn all_rows(num_rows: usize) -> Self {
        Self {
            indices: (0..num_rows as u32).collect(),
            weights: None,
        }
    }

    /// Get the weight for a row at the given index in the sample.
    #[inline]
    pub fn weight(&self, idx_in_sample: usize) -> f32 {
        self.weights
            .as_ref()
            .map(|w| w[idx_in_sample])
            .unwrap_or(1.0)
    }

    /// Check if this sample uses all rows (no filtering).
    #[inline]
    pub fn is_full(&self) -> bool {
        self.weights.is_none()
    }
}

// ============================================================================
// NoSampler (identity sampler)
// ============================================================================

/// No-op sampler that returns all rows.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoSampler;

impl RowSampler for NoSampler {
    fn sample(&self, num_rows: usize, _gradients: Option<&[f32]>, _seed: u64) -> RowSample {
        RowSample::all_rows(num_rows)
    }

    fn sample_multioutput(&self, grads: &GradientBuffer, _seed: u64) -> RowSample {
        RowSample::all_rows(grads.n_samples())
    }

    fn is_enabled(&self) -> bool {
        false
    }
}

// ============================================================================
// RandomSampler
// ============================================================================

/// Random row sampler without replacement.
///
/// Samples a fraction of rows uniformly at random. Does not use gradient
/// information - the same sampling applies regardless of gradients.
#[derive(Debug, Clone, Copy)]
pub struct RandomSampler {
    /// Fraction of rows to sample (0, 1].
    pub rate: f32,
}

impl RandomSampler {
    /// Create a new random sampler.
    ///
    /// # Panics
    /// Panics if `rate` is not in (0, 1].
    pub fn new(rate: f32) -> Self {
        assert!(
            rate > 0.0 && rate <= 1.0,
            "rate must be in (0, 1], got {}",
            rate
        );
        Self { rate }
    }

    /// Core sampling logic - samples `num_rows` randomly.
    fn do_sample(&self, num_rows: usize, seed: u64) -> RowSample {
        if self.rate >= 1.0 || num_rows == 0 {
            return RowSample::all_rows(num_rows);
        }

        let sample_size = ((num_rows as f32 * self.rate).ceil() as usize).max(1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

        // Sample without replacement using partial Fisher-Yates shuffle
        let mut indices: Vec<u32> = (0..num_rows as u32).collect();
        for i in 0..sample_size {
            let j = rng.gen_range(i..num_rows);
            indices.swap(i, j);
        }

        // Take first sample_size elements and sort for cache efficiency
        let mut sampled: Vec<u32> = indices[..sample_size].to_vec();
        sampled.sort_unstable();

        RowSample {
            indices: sampled,
            weights: None,
        }
    }
}

impl RowSampler for RandomSampler {
    fn sample(&self, num_rows: usize, _gradients: Option<&[f32]>, seed: u64) -> RowSample {
        self.do_sample(num_rows, seed)
    }

    fn sample_multioutput(&self, grads: &GradientBuffer, seed: u64) -> RowSample {
        // Random sampling doesn't use gradients - just sample by count
        self.do_sample(grads.n_samples(), seed)
    }

    fn is_enabled(&self) -> bool {
        self.rate < 1.0
    }
}

// ============================================================================
// GossSampler (Gradient-based One-Side Sampling)
// ============================================================================

/// Gradient-based One-Side Sampling (GOSS).
///
/// GOSS selects rows based on gradient magnitude:
/// 1. Keep top `top_rate` fraction by |gradient| (always included)
/// 2. Randomly sample `other_rate` fraction of remaining rows
/// 3. Apply weight amplification to sampled rows to compensate for bias
///
/// This focuses training on informative samples (large gradients) while
/// maintaining dataset distribution through weighted sampling.
///
/// For multi-output models, uses L2 norm of gradient vectors as the
/// per-row importance measure.
#[derive(Debug, Clone, Copy)]
pub struct GossSampler {
    /// Fraction of rows to keep by top gradient magnitude.
    pub top_rate: f32,
    /// Fraction of remaining rows to randomly sample.
    pub other_rate: f32,
}

impl Default for GossSampler {
    fn default() -> Self {
        Self {
            top_rate: 0.2,
            other_rate: 0.1,
        }
    }
}

impl GossSampler {
    /// Create a new GOSS sampler.
    ///
    /// # Panics
    /// Panics if `top_rate` or `other_rate` is not in (0, 1].
    pub fn new(top_rate: f32, other_rate: f32) -> Self {
        assert!(
            top_rate > 0.0 && top_rate <= 1.0,
            "top_rate must be in (0, 1], got {}",
            top_rate
        );
        assert!(
            other_rate > 0.0 && other_rate <= 1.0,
            "other_rate must be in (0, 1], got {}",
            other_rate
        );
        Self { top_rate, other_rate }
    }

    /// Compute the weight amplification factor for sampled (non-top) rows.
    ///
    /// Formula: (1 - top_rate) / other_rate
    #[inline]
    pub fn weight_amplification(&self) -> f32 {
        (1.0 - self.top_rate) / self.other_rate
    }

    /// Returns true if GOSS would sample (not all rows kept).
    #[inline]
    fn filters_rows(&self) -> bool {
        self.top_rate + self.other_rate * (1.0 - self.top_rate) < 1.0
    }

    /// Core GOSS sampling from pre-computed gradient magnitudes.
    fn sample_from_magnitudes(&self, magnitudes: &[f32], seed: u64) -> RowSample {
        let num_rows = magnitudes.len();

        if !self.filters_rows() || num_rows == 0 {
            return RowSample::all_rows(num_rows);
        }

        // Compute absolute gradient magnitudes with indices
        let mut indexed_grads: Vec<(u32, f32)> = magnitudes
            .iter()
            .enumerate()
            .map(|(i, &g)| (i as u32, g.abs()))
            .collect();

        // Sort by magnitude (descending)
        indexed_grads.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate top row count
        let top_count = ((num_rows as f32 * self.top_rate).ceil() as usize).max(1);
        let top_count = top_count.min(num_rows);

        // Top rows with weight 1.0
        let mut result_indices: Vec<u32> = indexed_grads[..top_count]
            .iter()
            .map(|(i, _)| *i)
            .collect();
        let mut result_weights: Vec<f32> = vec![1.0; top_count];

        // Sample from remaining rows
        let remaining_count = num_rows - top_count;
        if remaining_count > 0 {
            let sample_count =
                ((remaining_count as f32 * self.other_rate).ceil() as usize).max(1);
            let sample_count = sample_count.min(remaining_count);

            let remaining_indices: Vec<u32> = indexed_grads[top_count..]
                .iter()
                .map(|(i, _)| *i)
                .collect();

            let sampled = sample_from_slice(&remaining_indices, sample_count, seed);
            let weight = self.weight_amplification();

            for idx in sampled {
                result_indices.push(idx);
                result_weights.push(weight);
            }
        }

        // Sort for cache-friendly access
        let mut pairs: Vec<(u32, f32)> = result_indices
            .into_iter()
            .zip(result_weights)
            .collect();
        pairs.sort_by_key(|(i, _)| *i);

        RowSample {
            indices: pairs.iter().map(|(i, _)| *i).collect(),
            weights: Some(pairs.iter().map(|(_, w)| *w).collect()),
        }
    }
}

impl RowSampler for GossSampler {
    fn sample(&self, num_rows: usize, gradients: Option<&[f32]>, seed: u64) -> RowSample {
        if !self.filters_rows() || num_rows == 0 {
            return RowSample::all_rows(num_rows);
        }

        let grads = gradients.expect("GOSS requires gradients");
        assert_eq!(grads.len(), num_rows, "gradient length must match num_rows");

        self.sample_from_magnitudes(grads, seed)
    }

    fn sample_multioutput(&self, grads: &GradientBuffer, seed: u64) -> RowSample {
        let num_rows = grads.n_samples();
        let num_outputs = grads.n_outputs();

        if !self.filters_rows() || num_rows == 0 {
            return RowSample::all_rows(num_rows);
        }

        // For single output, extract gradients directly
        if num_outputs == 1 {
            let gradients: Vec<f32> = (0..num_rows).map(|i| grads.get(i, 0).0).collect();
            return self.sample_from_magnitudes(&gradients, seed);
        }

        // For multi-output, compute L2 norm of gradient vectors
        let magnitudes: Vec<f32> = (0..num_rows)
            .map(|row| {
                let mut sum_sq = 0.0f32;
                for k in 0..num_outputs {
                    let (grad, _) = grads.get(row, k);
                    sum_sq += grad * grad;
                }
                sum_sq.sqrt()
            })
            .collect();

        self.sample_from_magnitudes(&magnitudes, seed)
    }

    fn is_enabled(&self) -> bool {
        self.filters_rows()
    }
}

// ============================================================================
// RowSampling (unified enum)
// ============================================================================

/// Row sampling strategy configuration.
///
/// This enum provides a unified interface for all row sampling strategies.
/// It implements [`RowSampler`] by delegating to the appropriate concrete sampler.
///
/// # Example
///
/// ```ignore
/// use booste_rs::training::{GBTreeTrainer, gbtree::RowSampling};
///
/// // 80% random subsample
/// let trainer = GBTreeTrainer::builder()
///     .row_sampling(RowSampling::Random { rate: 0.8 })
///     .build()
///     .unwrap();
///
/// // GOSS sampling
/// let trainer = GBTreeTrainer::builder()
///     .row_sampling(RowSampling::goss_default())
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RowSampling {
    /// No row sampling - use all rows (default).
    None,
    /// Random subsample without replacement.
    Random {
        /// Fraction of rows to sample (0, 1].
        rate: f32,
    },
    /// Gradient-based One-Side Sampling (GOSS).
    Goss {
        /// Fraction of rows to keep by top gradient magnitude.
        top_rate: f32,
        /// Fraction of remaining rows to randomly sample.
        other_rate: f32,
    },
}

impl Default for RowSampling {
    fn default() -> Self {
        Self::None
    }
}

impl RowSampling {
    /// Create GOSS sampling with default parameters (top_rate=0.2, other_rate=0.1).
    pub fn goss_default() -> Self {
        Self::Goss {
            top_rate: 0.2,
            other_rate: 0.1,
        }
    }
}

impl fmt::Display for RowSampling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Random { rate } => write!(f, "Random(rate={:.2})", rate),
            Self::Goss { top_rate, other_rate } => {
                write!(f, "GOSS(top={:.2}, other={:.2})", top_rate, other_rate)
            }
        }
    }
}

impl RowSampler for RowSampling {
    fn sample(&self, num_rows: usize, gradients: Option<&[f32]>, seed: u64) -> RowSample {
        match self {
            Self::None => NoSampler.sample(num_rows, gradients, seed),
            Self::Random { rate } => RandomSampler::new(*rate).sample(num_rows, gradients, seed),
            Self::Goss { top_rate, other_rate } => {
                GossSampler::new(*top_rate, *other_rate).sample(num_rows, gradients, seed)
            }
        }
    }

    fn sample_multioutput(&self, grads: &GradientBuffer, seed: u64) -> RowSample {
        match self {
            Self::None => NoSampler.sample_multioutput(grads, seed),
            Self::Random { rate } => RandomSampler::new(*rate).sample_multioutput(grads, seed),
            Self::Goss { top_rate, other_rate } => {
                GossSampler::new(*top_rate, *other_rate).sample_multioutput(grads, seed)
            }
        }
    }

    fn is_enabled(&self) -> bool {
        match self {
            Self::None => false,
            Self::Random { rate } => *rate < 1.0,
            Self::Goss { top_rate, other_rate } => {
                top_rate + other_rate * (1.0 - top_rate) < 1.0
            }
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Sample `k` items from a slice without replacement.
///
/// Returns sorted values.
fn sample_from_slice(items: &[u32], k: usize, seed: u64) -> Vec<u32> {
    if k >= items.len() {
        return items.to_vec();
    }

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..items.len()).collect();

    for i in 0..k {
        let j = rng.gen_range(i..items.len());
        indices.swap(i, j);
    }

    let mut sampled: Vec<u32> = indices[..k].iter().map(|&i| items[i]).collect();
    sampled.sort_unstable();
    sampled
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- RowSampling Tests ----

    #[test]
    fn test_row_sampling_none() {
        let config = RowSampling::None;
        assert!(!config.is_enabled());
        assert_eq!(format!("{}", config), "None");
    }

    #[test]
    fn test_row_sampling_random() {
        let config = RowSampling::Random { rate: 0.8 };
        assert!(config.is_enabled());
        assert!(format!("{}", config).contains("Random"));
    }

    #[test]
    fn test_row_sampling_goss() {
        let config = RowSampling::goss_default();
        assert!(config.is_enabled());
        assert!(format!("{}", config).contains("GOSS"));
    }

    // ---- RowSampler trait tests via RowSampling ----

    #[test]
    fn test_row_sampling_sample_none() {
        let config = RowSampling::None;
        let sample = config.sample(100, None, 42);
        assert_eq!(sample.indices.len(), 100);
        assert_eq!(sample.indices, (0..100).collect::<Vec<_>>());
        assert!(sample.weights.is_none());
    }

    #[test]
    fn test_row_sampling_sample_random() {
        let config = RowSampling::Random { rate: 0.5 };
        let sample = config.sample(100, None, 42);
        assert_eq!(sample.indices.len(), 50);

        // Should be sorted
        for i in 1..sample.indices.len() {
            assert!(sample.indices[i] > sample.indices[i - 1]);
        }

        // All indices should be valid
        for &idx in &sample.indices {
            assert!(idx < 100);
        }

        // No weights for random sampling
        assert!(sample.weights.is_none());
    }

    #[test]
    fn test_row_sampling_sample_goss() {
        let config = RowSampling::Goss {
            top_rate: 0.3,
            other_rate: 0.2,
        };
        let gradients: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let sample = config.sample(10, Some(&gradients), 42);

        // Should have top 3 + sampled from remaining
        assert!(sample.indices.len() >= 3);
        assert!(sample.weights.is_some());

        // Top rows (7, 8, 9) have weight 1.0
        let weights = sample.weights.as_ref().unwrap();
        let top_rows = vec![7u32, 8, 9];
        for (i, &idx) in sample.indices.iter().enumerate() {
            if top_rows.contains(&idx) {
                assert_eq!(weights[i], 1.0);
            }
        }
    }

    #[test]
    fn test_row_sampling_reproducible() {
        let config = RowSampling::Random { rate: 0.5 };

        let sample1 = config.sample(100, None, 42);
        let sample2 = config.sample(100, None, 42);

        assert_eq!(sample1.indices, sample2.indices);
    }

    #[test]
    fn test_row_sampling_different_seeds() {
        let config = RowSampling::Random { rate: 0.5 };

        let sample1 = config.sample(100, None, 42);
        let sample2 = config.sample(100, None, 123);

        assert_ne!(sample1.indices, sample2.indices);
    }

    // ---- RandomSampler Tests ----

    #[test]
    fn test_random_sampler_basic() {
        let sampler = RandomSampler::new(0.5);
        assert!(sampler.is_enabled());

        let sample = sampler.sample(100, None, 42);
        assert_eq!(sample.indices.len(), 50);
        assert!(sample.weights.is_none());
    }

    #[test]
    fn test_random_sampler_full() {
        let sampler = RandomSampler::new(1.0);
        assert!(!sampler.is_enabled());

        let sample = sampler.sample(100, None, 42);
        assert_eq!(sample.indices.len(), 100);
    }

    #[test]
    fn test_random_sampler_multioutput_ignores_gradients() {
        let sampler = RandomSampler::new(0.5);

        // Create gradient buffer with different values
        let mut grads = GradientBuffer::new(100, 3);
        for row in 0..100 {
            for k in 0..3 {
                grads.set(row, k, row as f32, 1.0);
            }
        }

        let sample = sampler.sample_multioutput(&grads, 42);

        // Should still sample 50 rows regardless of gradient values
        assert_eq!(sample.indices.len(), 50);
        assert!(sample.weights.is_none());
    }

    #[test]
    #[should_panic(expected = "rate must be in (0, 1]")]
    fn test_random_sampler_invalid_zero() {
        RandomSampler::new(0.0);
    }

    // ---- GossSampler Tests ----

    #[test]
    fn test_goss_sampler_default() {
        let sampler = GossSampler::default();
        assert_eq!(sampler.top_rate, 0.2);
        assert_eq!(sampler.other_rate, 0.1);
        assert!(sampler.is_enabled());
    }

    #[test]
    fn test_goss_sampler_not_enabled() {
        // When top_rate + other_rate * (1 - top_rate) >= 1.0, all rows are kept
        let sampler = GossSampler::new(0.5, 1.0); // 0.5 + 1.0 * 0.5 = 1.0
        assert!(!sampler.is_enabled());
    }

    #[test]
    fn test_goss_sampler_weight_amplification() {
        let sampler = GossSampler::new(0.2, 0.1);
        // Weight = (1 - 0.2) / 0.1 = 0.8 / 0.1 = 8.0
        assert!((sampler.weight_amplification() - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_goss_sampler_all_rows() {
        let sampler = GossSampler::new(0.5, 1.0);

        let gradients: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let sample = sampler.sample(10, Some(&gradients), 42);

        // Should include all rows
        assert_eq!(sample.indices.len(), 10);
        assert_eq!(sample.indices, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_goss_sampler_basic() {
        let sampler = GossSampler::new(0.3, 0.2);

        // 10 rows with varying gradients
        let gradients: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let sample = sampler.sample(10, Some(&gradients), 42);

        // Top 3 (30%) are indices 9, 8, 7 (highest absolute gradients)
        let top_rows = vec![7u32, 8, 9];
        for &top in &top_rows {
            assert!(sample.indices.contains(&top), "Top row {} missing", top);
        }

        // Total: 3 top + ceil(7 * 0.2) = 5 rows
        assert_eq!(sample.indices.len(), 5);

        // Check weights
        let weights = sample.weights.as_ref().unwrap();
        for (i, &idx) in sample.indices.iter().enumerate() {
            if top_rows.contains(&idx) {
                assert_eq!(weights[i], 1.0, "Top row should have weight 1.0");
            } else {
                let expected_weight = sampler.weight_amplification();
                assert!(
                    (weights[i] - expected_weight).abs() < 1e-6,
                    "Sampled row should have weight {}, got {}",
                    expected_weight,
                    weights[i]
                );
            }
        }
    }

    #[test]
    fn test_goss_sampler_sorted_indices() {
        let sampler = GossSampler::new(0.2, 0.3);

        let gradients: Vec<f32> = (0..100).map(|i| (i as f32).sin()).collect();
        let sample = sampler.sample(100, Some(&gradients), 42);

        // Indices should be sorted for cache efficiency
        for i in 1..sample.indices.len() {
            assert!(
                sample.indices[i] > sample.indices[i - 1],
                "Indices should be sorted"
            );
        }
    }

    #[test]
    fn test_goss_sampler_reproducible() {
        let sampler = GossSampler::new(0.2, 0.1);

        let gradients: Vec<f32> = (0..50).map(|i| (i as f32).sin()).collect();

        let sample1 = sampler.sample(50, Some(&gradients), 42);
        let sample2 = sampler.sample(50, Some(&gradients), 42);

        assert_eq!(sample1.indices, sample2.indices);
    }

    #[test]
    fn test_goss_sampler_different_seeds() {
        let sampler = GossSampler::new(0.2, 0.1);

        let gradients: Vec<f32> = (0..50).map(|i| (i as f32).sin()).collect();

        let sample1 = sampler.sample(50, Some(&gradients), 42);
        let sample2 = sampler.sample(50, Some(&gradients), 123);

        assert_ne!(sample1.indices, sample2.indices);
    }

    #[test]
    fn test_goss_sampler_negative_gradients() {
        let sampler = GossSampler::new(0.3, 0.2);

        // Mix of positive and negative gradients
        let gradients: Vec<f32> = vec![-9.0, 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, -7.0, 8.0, 0.5];
        let sample = sampler.sample(10, Some(&gradients), 42);

        // Top 3 by absolute value: indices 0 (|-9|), 8 (|8|), 7 (|-7|)
        assert!(sample.indices.contains(&0), "Index 0 (grad=-9) should be top");
        assert!(sample.indices.contains(&8), "Index 8 (grad=8) should be top");
        assert!(sample.indices.contains(&7), "Index 7 (grad=-7) should be top");
    }

    #[test]
    #[should_panic(expected = "top_rate must be in (0, 1]")]
    fn test_goss_sampler_invalid_top_rate() {
        GossSampler::new(0.0, 0.1);
    }

    #[test]
    #[should_panic(expected = "other_rate must be in (0, 1]")]
    fn test_goss_sampler_invalid_other_rate() {
        GossSampler::new(0.2, 0.0);
    }

    // ---- GOSS Multi-output Tests ----

    #[test]
    fn test_goss_multioutput_basic() {
        let sampler = GossSampler::new(0.3, 0.2);

        // Create 10 rows × 3 outputs
        let mut grads = GradientBuffer::new(10, 3);

        // Set gradients so rows 0, 1, 2 have highest L2 norms
        for row in 0..10 {
            let grad = if row < 3 { (9 - row) as f32 } else { 0.5 };
            for k in 0..3 {
                grads.set(row, k, grad, 1.0);
            }
        }

        let sample = sampler.sample_multioutput(&grads, 42);

        // Top 3 rows (30% of 10) should always be included
        assert!(sample.indices.contains(&0), "Row 0 should be top");
        assert!(sample.indices.contains(&1), "Row 1 should be top");
        assert!(sample.indices.contains(&2), "Row 2 should be top");

        // Sample should have 3 top + ceil(7 * 0.2) = 5 rows
        assert_eq!(sample.indices.len(), 5);
    }

    #[test]
    fn test_goss_multioutput_uses_l2_norm() {
        let sampler = GossSampler::new(0.2, 0.1);

        // 5 rows × 2 outputs
        let mut grads = GradientBuffer::new(5, 2);

        // Row 0: [10, 0] → L2 = 10
        // Row 1: [0, 10] → L2 = 10
        // Row 2: [7, 7] → L2 = sqrt(98) ≈ 9.9
        // Row 3: [1, 1] → L2 = sqrt(2) ≈ 1.4
        // Row 4: [0, 0] → L2 = 0
        grads.set(0, 0, 10.0, 1.0);
        grads.set(0, 1, 0.0, 1.0);
        grads.set(1, 0, 0.0, 1.0);
        grads.set(1, 1, 10.0, 1.0);
        grads.set(2, 0, 7.0, 1.0);
        grads.set(2, 1, 7.0, 1.0);
        grads.set(3, 0, 1.0, 1.0);
        grads.set(3, 1, 1.0, 1.0);
        grads.set(4, 0, 0.0, 1.0);
        grads.set(4, 1, 0.0, 1.0);

        let sample = sampler.sample_multioutput(&grads, 42);

        // One of the high-L2 rows should be in top
        assert!(
            sample.indices.contains(&0) || sample.indices.contains(&1) || sample.indices.contains(&2),
            "One of the high-L2 rows should be in top"
        );
    }

    #[test]
    fn test_goss_multioutput_delegates_to_single_output() {
        let sampler = GossSampler::new(0.2, 0.1);

        // Single-output case
        let mut grads = GradientBuffer::new(50, 1);
        for row in 0..50 {
            grads.set(row, 0, (row as f32).sin(), 1.0);
        }

        let sample_multi = sampler.sample_multioutput(&grads, 42);

        // Compare with direct single-output call
        let gradients: Vec<f32> = (0..50).map(|i| grads.get(i, 0).0).collect();
        let sample_single = sampler.sample(50, Some(&gradients), 42);

        assert_eq!(sample_multi.indices, sample_single.indices);
    }
}
