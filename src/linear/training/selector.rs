//! Feature selectors for coordinate descent.
//!
//! Controls the order in which features are updated during training.

use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Trait for selecting features during coordinate descent.
pub trait FeatureSelector: Send + Sync {
    /// Reset the selector for a new round.
    fn reset(&mut self, num_features: usize);

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
    num_features: usize,
    current: usize,
}

impl CyclicSelector {
    /// Create a new cyclic selector.
    pub fn new() -> Self {
        Self::default()
    }
}

impl FeatureSelector for CyclicSelector {
    fn reset(&mut self, num_features: usize) {
        self.num_features = num_features;
        self.current = 0;
    }

    fn next(&mut self) -> Option<usize> {
        if self.current < self.num_features {
            let idx = self.current;
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }

    fn all_indices(&mut self) -> Vec<usize> {
        self.current = self.num_features;
        (0..self.num_features).collect()
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
    fn reset(&mut self, num_features: usize) {
        self.indices = (0..num_features).collect();
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

        let mut visited = vec![false; 5];
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
}
