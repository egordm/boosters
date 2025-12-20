//! Shared parallelism configuration.
//!
//! Provides a single [`Parallelism`] enum used throughout training. Components
//! receive a parallelism hint and self-correct if parallel execution would be
//! overkill for their workload.

/// Parallelism strategy for training operations.
///
/// This is a *hint* that components can override. For example, histogram building
/// may choose sequential execution even when `Parallel(8)` is specified if the
/// workload is too small to benefit from parallelism.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parallelism {
    /// Force strictly sequential execution (no thread spawning).
    Sequential,
    /// Allow parallel execution with up to `n` threads.
    ///
    /// If `n <= 1`, this is equivalent to `Sequential`.
    Parallel(usize),
}

impl Default for Parallelism {
    fn default() -> Self {
        Self::Sequential
    }
}

impl Parallelism {
    /// Create a parallelism hint from a thread count.
    ///
    /// - `0` → uses rayon's current thread count
    /// - `1` → sequential
    /// - `n > 1` → parallel with n threads
    #[inline]
    pub fn from_threads(n_threads: usize) -> Self {
        match n_threads {
            0 => Self::Parallel(rayon::current_num_threads()),
            1 => Self::Sequential,
            n => Self::Parallel(n),
        }
    }

    /// Returns `true` if parallel execution is allowed.
    #[inline]
    pub fn allows_parallel(self) -> bool {
        matches!(self, Self::Parallel(n) if n > 1)
    }

    /// Returns the thread count hint (1 for sequential).
    #[inline]
    pub fn n_threads(self) -> usize {
        match self {
            Self::Sequential => 1,
            Self::Parallel(n) => n.max(1),
        }
    }

    /// Self-correct: if the workload is too small, downgrade to sequential.
    ///
    /// This is the "correction" mechanism: algorithms call this with their
    /// workload characteristics and get back the actual strategy to use.
    #[inline]
    pub fn correct_for_workload(self, n_items: usize, min_items_per_thread: usize) -> Self {
        match self {
            Self::Sequential => Self::Sequential,
            Self::Parallel(n) => {
                let effective_threads = n.min(n_items / min_items_per_thread.max(1)).max(1);
                if effective_threads <= 1 {
                    Self::Sequential
                } else {
                    Self::Parallel(effective_threads)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_threads() {
        assert_eq!(Parallelism::from_threads(1), Parallelism::Sequential);
        assert_eq!(Parallelism::from_threads(4), Parallelism::Parallel(4));
        // n_threads=0 uses rayon's current count, which varies
        assert!(matches!(Parallelism::from_threads(0), Parallelism::Parallel(_)));
    }

    #[test]
    fn test_allows_parallel() {
        assert!(!Parallelism::Sequential.allows_parallel());
        assert!(!Parallelism::Parallel(1).allows_parallel());
        assert!(Parallelism::Parallel(2).allows_parallel());
        assert!(Parallelism::Parallel(8).allows_parallel());
    }

    #[test]
    fn test_n_threads() {
        assert_eq!(Parallelism::Sequential.n_threads(), 1);
        assert_eq!(Parallelism::Parallel(4).n_threads(), 4);
        assert_eq!(Parallelism::Parallel(0).n_threads(), 1); // edge case
    }

    #[test]
    fn test_correct_for_workload() {
        // Sequential stays sequential
        assert_eq!(
            Parallelism::Sequential.correct_for_workload(1000, 100),
            Parallelism::Sequential
        );

        // Small workload → sequential
        assert_eq!(
            Parallelism::Parallel(8).correct_for_workload(100, 100),
            Parallelism::Sequential
        );

        // Large workload → parallel (possibly reduced thread count)
        let corrected = Parallelism::Parallel(8).correct_for_workload(400, 100);
        assert!(matches!(corrected, Parallelism::Parallel(n) if n >= 2 && n <= 4));

        // Very large workload → full parallelism
        assert_eq!(
            Parallelism::Parallel(8).correct_for_workload(10000, 100),
            Parallelism::Parallel(8)
        );
    }
}
