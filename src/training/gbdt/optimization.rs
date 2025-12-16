//! Optimization profiles for automatic strategy selection.
//!
//! This module provides automatic selection of parallelism strategies based on
//! data characteristics. The main entry point is [`OptimizationProfile`], which
//! resolves to a [`Parallelism`] hint that algorithms can further self-correct
//! based on their specific workloads.

use super::parallelism::Parallelism;

/// Minimum features to enable parallel split finding by default.
const MIN_FEATURES_PARALLEL: usize = 16;

// =============================================================================
// Optimization Profile
// =============================================================================

/// High-level optimization profile for parallelism strategy.
///
/// Users select a profile based on their data characteristics or use `Auto`
/// for automatic detection. The resolved `Parallelism` is a hint that algorithms
/// may further self-correct based on their specific workload (e.g., number of
/// features to iterate).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OptimizationProfile {
    /// Auto-detect based on data characteristics.
    #[default]
    Auto,
    /// Optimized for small datasets (<100k rows). Sequential strategies.
    SmallData,
    /// Optimized for medium/large datasets (>=100k rows). Parallel strategies.
    LargeData,
}

impl OptimizationProfile {
    /// Resolve profile to parallelism hint.
    ///
    /// The returned `Parallelism` is a hint that algorithms can further
    /// self-correct based on their specific workload.
    ///
    /// # Arguments
    /// * `n_rows` - Number of samples
    /// * `n_features` - Number of features
    pub fn resolve(&self, n_rows: usize, n_features: usize) -> Parallelism {
        match self {
            Self::Auto => Self::auto_detect(n_rows, n_features),
            Self::SmallData => Parallelism::Sequential,
            Self::LargeData => {
                if n_features >= MIN_FEATURES_PARALLEL {
                    // Use rayon's default thread count
                    Parallelism::from_threads(0)
                } else {
                    Parallelism::Sequential
                }
            }
        }
    }

    /// Auto-detect optimal parallelism from data characteristics.
    fn auto_detect(n_rows: usize, n_features: usize) -> Parallelism {
        // Use parallel strategies for larger datasets with enough features
        if n_rows >= 100_000 && n_features >= MIN_FEATURES_PARALLEL {
            Parallelism::from_threads(0)
        } else {
            Parallelism::Sequential
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_detect_small() {
        let parallelism = OptimizationProfile::Auto.resolve(10_000, 50);
        assert_eq!(parallelism, Parallelism::Sequential);
    }

    #[test]
    fn test_auto_detect_medium() {
        let parallelism = OptimizationProfile::Auto.resolve(500_000, 100);
        // Should return parallel (thread count varies)
        assert!(parallelism.allows_parallel());
    }

    #[test]
    fn test_auto_detect_few_features() {
        // Even large dataset should use sequential if few features
        let parallelism = OptimizationProfile::Auto.resolve(1_000_000, 5);
        assert_eq!(parallelism, Parallelism::Sequential);
    }

    #[test]
    fn test_large_data_profile() {
        let parallelism = OptimizationProfile::LargeData.resolve(10_000, 100);
        assert!(parallelism.allows_parallel());
    }

    #[test]
    fn test_small_data_profile() {
        let parallelism = OptimizationProfile::SmallData.resolve(1_000_000, 100);
        assert_eq!(parallelism, Parallelism::Sequential);
    }
}

