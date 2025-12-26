//! Row sampling strategies for gradient boosting.
//!
//! This module provides different row sampling strategies that can be used
//! during training to reduce computation and improve generalization.
//!
//! # Strategies
//!
//! - **None**: Use all rows (no sampling)
//! - **Uniform**: Random uniform subsampling (bagging)
//! - **GOSS**: Gradient-based One-Side Sampling - keeps top gradients, samples rest
//!
//! # GOSS Algorithm
//!
//! GOSS (Gradient-based One-Side Sampling) from LightGBM:
//! 1. Compute gradient magnitude for each sample: `|grad * hess|`
//! 2. Keep top `top_rate` fraction of samples (high gradient)
//! 3. Randomly sample `other_rate` fraction from the rest
//! 4. Multiply sampled small gradients by `(1 - top_rate) / other_rate` to correct bias
//!
//! This provides significant speedup while maintaining accuracy because:
//! - Samples with large gradients contribute more to information gain
//! - Samples with small gradients are already well-trained

use rand::prelude::*;
use rand::rngs::SmallRng;

use crate::training::GradsTuple;

/// Configuration for row sampling.
#[derive(Clone, Debug, Default)]
pub enum RowSamplingParams {
    /// No sampling - use all rows.
    #[default]
    None,
    /// Uniform random sampling (bagging).
    Uniform {
        /// Fraction of rows to sample (0, 1].
        subsample: f32,
    },
    /// Gradient-based One-Side Sampling.
    Goss {
        /// Fraction of top gradient samples to keep (e.g., 0.2).
        top_rate: f32,
        /// Fraction of remaining samples to randomly select (e.g., 0.1).
        other_rate: f32,
    },
}

impl RowSamplingParams {
    /// Create uniform sampling config.
    pub fn uniform(subsample: f32) -> Self {
        assert!(
            subsample > 0.0 && subsample <= 1.0,
            "subsample must be in (0, 1]"
        );
        if (subsample - 1.0).abs() < 1e-6 {
            Self::None
        } else {
            Self::Uniform { subsample }
        }
    }

    /// Create GOSS sampling config.
    pub fn goss(top_rate: f32, other_rate: f32) -> Self {
        assert!(top_rate > 0.0, "top_rate must be positive");
        assert!(other_rate > 0.0, "other_rate must be positive");
        assert!(
            top_rate + other_rate <= 1.0,
            "top_rate + other_rate must be <= 1.0"
        );
        Self::Goss {
            top_rate,
            other_rate,
        }
    }
}

/// Row sampler for selecting training samples each iteration.
pub struct RowSampler {
    /// Sampling configuration.
    config: RowSamplingParams,
    /// Random number generator.
    rng: SmallRng,
    /// Buffer for selected indices.
    indices: Vec<u32>,
    /// Reusable mask buffer for uniform sampling.
    ///
    /// Values are 0/1 and used to zero out gradients/hessians for unsampled rows
    /// without allocating a fresh `Vec<bool>` each iteration.
    uniform_mask: Vec<u8>,
    /// Buffer for gradient magnitudes (for GOSS).
    grad_magnitudes: Vec<f32>,
    /// Number of warmup iterations before GOSS kicks in.
    warmup_rounds: usize,
}

impl RowSampler {
    /// Create a new row sampler.
    ///
    /// # Arguments
    /// * `config` - Sampling configuration
    /// * `n_rows` - Total number of rows in dataset
    /// * `seed` - Random seed
    /// * `learning_rate` - Learning rate (used to compute GOSS warmup)
    pub fn new(config: RowSamplingParams, n_rows: usize, seed: u64, learning_rate: f32) -> Self {
        let rng = SmallRng::seed_from_u64(seed);

        // GOSS warmup: don't sample for first 1/learning_rate iterations
        let warmup_rounds = match &config {
            RowSamplingParams::Goss { .. } => (1.0 / learning_rate) as usize,
            _ => 0,
        };

        let grad_magnitudes = match &config {
            RowSamplingParams::Goss { .. } => Vec::with_capacity(n_rows),
            _ => Vec::new(),
        };

        Self {
            config,
            rng,
            indices: Vec::with_capacity(n_rows),
            uniform_mask: vec![0; n_rows],
            grad_magnitudes,
            warmup_rounds,
        }
    }

    /// Sample rows for the current iteration.
    ///
    /// Returns `None` if no sampling is needed (use all rows).
    /// Returns `Some(&[u32])` with selected row indices if sampling is active.
    ///
    /// # Arguments
    /// * `iteration` - Current boosting iteration (0-indexed)
    /// * `grad_hess` - Gradient/hessian pairs for all rows (modified in-place for GOSS)
    ///
    /// # Note
    /// For GOSS, the gradients and hessians are modified in-place to apply
    /// the amplification factor for sampled small-gradient samples.
    pub fn sample(&mut self, iteration: usize, grad_hess: &mut [GradsTuple]) -> Option<&[u32]> {
        let n_rows = grad_hess.len();

        match &self.config {
            RowSamplingParams::None => None,

            RowSamplingParams::Uniform { subsample } => {
                self.sample_uniform(grad_hess, n_rows, *subsample);
                Some(&self.indices)
            }

            RowSamplingParams::Goss {
                top_rate,
                other_rate,
            } => {
                // Skip GOSS during warmup
                if iteration < self.warmup_rounds {
                    return None;
                }

                self.sample_goss(grad_hess, *top_rate, *other_rate);
                Some(&self.indices)
            }
        }
    }

    /// Uniform random sampling.
    ///
    /// Zeros out gradients for unsampled rows to exclude them from tree building.
    fn sample_uniform(&mut self, grad_hess: &mut [GradsTuple], n_rows: usize, subsample: f32) {
        self.indices.clear();

        let target_count = ((n_rows as f32) * subsample) as usize;
        let target_count = target_count.max(1);

        let target_count = target_count.min(n_rows);
        if target_count == 0 {
            grad_hess.fill(GradsTuple::default());
            return;
        }

        // Reservoir sampling into `indices`.
        self.indices.extend(0..target_count as u32);
        for i in target_count..n_rows {
            let j = self.rng.gen_range(0..=i);
            if j < target_count {
                self.indices[j] = i as u32;
            }
        }

        // Zero out gradients for unsampled rows using a reusable mask.
        if self.uniform_mask.len() != n_rows {
            self.uniform_mask.resize(n_rows, 0);
        }
        self.uniform_mask.fill(0);
        for &row in &self.indices {
            self.uniform_mask[row as usize] = 1;
        }

        for (i, gh) in grad_hess.iter_mut().enumerate() {
            if self.uniform_mask[i] == 0 {
                gh.grad = 0.0;
                gh.hess = 0.0;
            }
        }

        self.indices.sort_unstable();
    }

    /// GOSS sampling.
    fn sample_goss(&mut self, grad_hess: &mut [GradsTuple], top_rate: f32, other_rate: f32) {
        let n_rows = grad_hess.len();

        // Step 1: Compute gradient magnitudes |grad * hess|
        self.grad_magnitudes.clear();
        self.grad_magnitudes
            .extend(grad_hess.iter().map(|gh| (gh.grad * gh.hess).abs()));

        // Step 2: Find threshold using partial sort (quickselect)
        let top_k = ((n_rows as f32) * top_rate) as usize;
        let top_k = top_k.max(1).min(n_rows - 1);
        let other_k = ((n_rows as f32) * other_rate) as usize;
        let other_k = other_k.max(1);

        // Clone for partitioning (we need original order)
        let mut sorted_mags = self.grad_magnitudes.clone();
        let threshold_idx = n_rows - top_k;
        let (_, threshold_val, _) = sorted_mags.select_nth_unstable_by(threshold_idx, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let threshold = *threshold_val;

        // Step 3: Select samples
        self.indices.clear();

        // Amplification factor for small gradients
        let multiply = (n_rows - top_k) as f32 / other_k as f32;

        let mut big_count = 0usize;
        let mut small_sampled = 0usize;

        for (i, gh) in grad_hess.iter_mut().enumerate() {
            let mag = self.grad_magnitudes[i];

            if mag >= threshold {
                // Top gradient - always keep
                self.indices.push(i as u32);
                big_count += 1;
            } else {
                // Small gradient - sample with adaptive probability
                let rest_need = other_k.saturating_sub(small_sampled);
                let rest_all = (n_rows - i).saturating_sub(top_k.saturating_sub(big_count));

                if rest_all > 0 {
                    let prob = rest_need as f32 / rest_all as f32;
                    if self.rng.r#gen::<f32>() < prob {
                        self.indices.push(i as u32);
                        small_sampled += 1;

                        // Amplify gradients to correct bias
                        gh.grad *= multiply;
                        gh.hess *= multiply;
                    }
                }
            }
        }

        // Sort for cache-friendly access
        self.indices.sort_unstable();
    }

    /// Get the number of samples that will be used.
    pub fn sample_count(&self, n_rows: usize, iteration: usize) -> usize {
        match &self.config {
            RowSamplingParams::None => n_rows,
            RowSamplingParams::Uniform { subsample } => ((n_rows as f32) * subsample) as usize,
            RowSamplingParams::Goss {
                top_rate,
                other_rate,
            } => {
                if iteration < self.warmup_rounds {
                    n_rows
                } else {
                    ((n_rows as f32) * (top_rate + other_rate)) as usize
                }
            }
        }
    }

    /// Check if sampling is active for this iteration.
    pub fn is_sampling_active(&self, iteration: usize) -> bool {
        match &self.config {
            RowSamplingParams::None => false,
            RowSamplingParams::Uniform { .. } => true,
            RowSamplingParams::Goss { .. } => iteration >= self.warmup_rounds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_sampling() {
        let mut sampler = RowSampler::new(RowSamplingParams::uniform(0.5), 100, 42, 0.1);

        let mut grad_hess = vec![
            GradsTuple {
                grad: 1.0,
                hess: 1.0
            };
            100
        ];

        let indices = sampler.sample(0, &mut grad_hess).unwrap();

        // Should sample approximately 50 rows
        assert!(indices.len() >= 40 && indices.len() <= 60);

        // Indices should be sorted
        for i in 1..indices.len() {
            assert!(indices[i] > indices[i - 1]);
        }
    }

    #[test]
    fn test_goss_warmup() {
        let mut sampler = RowSampler::new(
            RowSamplingParams::goss(0.2, 0.1),
            100,
            42,
            0.1, // warmup = 10 rounds
        );

        let mut grad_hess = vec![
            GradsTuple {
                grad: 1.0,
                hess: 1.0
            };
            100
        ];

        // During warmup, should return None
        assert!(sampler.sample(0, &mut grad_hess).is_none());
        assert!(sampler.sample(5, &mut grad_hess).is_none());
        assert!(sampler.sample(9, &mut grad_hess).is_none());

        // After warmup, should return indices
        assert!(sampler.sample(10, &mut grad_hess).is_some());
    }

    #[test]
    fn test_goss_sampling() {
        let n_rows = 1000;
        let mut sampler = RowSampler::new(
            RowSamplingParams::goss(0.2, 0.1),
            n_rows,
            42,
            100.0, // high lr = no warmup (warmup < 1)
        );

        // Create gradients with varying magnitudes
        let mut grad_hess: Vec<GradsTuple> = (0..n_rows)
            .map(|i| GradsTuple {
                grad: (i as f32) * 0.01,
                hess: 1.0,
            })
            .collect();

        let indices = sampler.sample(0, &mut grad_hess).unwrap();

        // Should sample approximately 30% (20% top + 10% other)
        let expected = (n_rows as f32 * 0.3) as usize;
        assert!(indices.len() >= expected - 50 && indices.len() <= expected + 50);

        // High gradient samples should be included
        // Last 200 samples (top 20%) should mostly be in indices
        let top_count = indices.iter().filter(|&&i| i >= 800).count();
        assert!(
            top_count >= 180,
            "Expected most top gradients to be kept, got {}",
            top_count
        );

        // Check that some small gradients were amplified
        let amplified_count = grad_hess.iter().filter(|gh| gh.grad > 1.0).count();
        assert!(
            amplified_count > 0,
            "Expected some gradients to be amplified"
        );
    }

    #[test]
    fn test_goss_amplification() {
        let n_rows = 100;
        let mut sampler = RowSampler::new(
            RowSamplingParams::goss(0.2, 0.1),
            n_rows,
            42,
            100.0, // high lr = no warmup
        );

        // All small gradients
        let mut grad_hess = vec![
            GradsTuple {
                grad: 0.01,
                hess: 1.0
            };
            n_rows
        ];

        // Set a few to be "top" gradients
        for gh in &mut grad_hess[80..100] {
            gh.grad = 1.0;
        }

        let original_small_grad = grad_hess[0].grad;

        sampler.sample(0, &mut grad_hess).unwrap();

        // Amplification factor should be (100 - 20) / 10 = 8.0
        // Some small gradients should be amplified
        let amplified: Vec<_> = grad_hess
            .iter()
            .take(80)
            .map(|gh| gh.grad)
            .filter(|&g| (g - original_small_grad).abs() > 0.001)
            .collect();

        if !amplified.is_empty() {
            let expected_amplified = original_small_grad * 8.0;
            assert!(
                (amplified[0] - expected_amplified).abs() < 0.01,
                "Expected amplification ~{}, got {}",
                expected_amplified,
                amplified[0]
            );
        }
    }

    #[test]
    fn test_no_sampling() {
        let mut sampler = RowSampler::new(RowSamplingParams::None, 100, 42, 0.1);

        let mut grad_hess = vec![
            GradsTuple {
                grad: 1.0,
                hess: 1.0
            };
            100
        ];

        assert!(sampler.sample(0, &mut grad_hess).is_none());
        assert!(!sampler.is_sampling_active(0));
    }

    #[test]
    fn test_sample_count() {
        let sampler = RowSampler::new(
            RowSamplingParams::goss(0.2, 0.1),
            1000,
            42,
            0.1, // warmup = 10
        );

        // During warmup
        assert_eq!(sampler.sample_count(1000, 0), 1000);

        // After warmup
        assert_eq!(sampler.sample_count(1000, 10), 300);
    }
}
