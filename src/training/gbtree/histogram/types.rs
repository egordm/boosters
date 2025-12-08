//! Core types for histogram pool and parallel building.
//!
//! This module defines the fundamental types used by the histogram pooling
//! and row-parallel building systems (RFC-0025).

use std::fmt;

/// Identifies a tree node for histogram caching.
///
/// Node IDs are assigned during tree building and used to index
/// histograms in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Create a new node ID.
    #[inline]
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the underlying ID value.
    #[inline]
    pub fn value(self) -> u32 {
        self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

impl From<u32> for NodeId {
    #[inline]
    fn from(id: u32) -> Self {
        Self(id)
    }
}

/// Internal slot identifier within the histogram pool.
///
/// Slots are indices into the contiguous backing store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SlotId(pub(crate) u32);

impl SlotId {
    #[inline]
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

/// Statistics for histogram pool usage.
///
/// Useful for tuning pool capacity and understanding access patterns.
#[derive(Debug, Clone, Default)]
pub struct PoolMetrics {
    /// Number of times a requested histogram was already in pool.
    pub hits: u64,
    /// Number of times a new slot had to be allocated.
    pub misses: u64,
    /// Number of times a slot was evicted via LRU.
    pub evictions: u64,
    /// Maximum number of slots in use simultaneously.
    pub peak_usage: usize,
}

impl PoolMetrics {
    /// Create new metrics with all counters at zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate hit rate as a percentage.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl fmt::Display for PoolMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PoolMetrics {{ hits: {}, misses: {}, evictions: {}, peak: {}, hit_rate: {:.1}% }}",
            self.hits,
            self.misses,
            self.evictions,
            self.peak_usage,
            self.hit_rate()
        )
    }
}

/// Calculate recommended pool capacity based on tree parameters.
///
/// # Strategy
///
/// - Base: 2 × max_leaves (need parent + both children during split)
/// - Buffer: +25% for histogram subtraction working space
/// - Minimum: 8 (even for very shallow trees)
/// - Cap: available memory / histogram_size
///
/// # Arguments
///
/// * `max_leaves` - Maximum number of leaves in the tree
/// * `max_depth` - Optional maximum tree depth (overrides max_leaves heuristic)
/// * `histogram_size_bytes` - Size of one histogram in bytes
/// * `available_memory_bytes` - Maximum memory budget for histograms
///
/// # Returns
///
/// Recommended number of histogram slots for the pool.
pub fn recommended_pool_capacity(
    max_leaves: usize,
    max_depth: Option<usize>,
    histogram_size_bytes: usize,
    available_memory_bytes: usize,
) -> usize {
    let base = if let Some(depth) = max_depth {
        // Depth-wise: need nodes at current + next level
        // At depth d, we can have up to 2^d nodes, plus 2^(d+1) children
        2usize.saturating_pow(depth as u32 + 1)
    } else {
        // Leaf-wise: need all candidate leaves + working space
        max_leaves.saturating_mul(2)
    };

    // Add 25% buffer for working space
    let with_buffer = (base as f64 * 1.25) as usize;

    // Cap by available memory
    let memory_cap = if histogram_size_bytes > 0 {
        available_memory_bytes / histogram_size_bytes
    } else {
        usize::MAX
    };

    // Enforce minimum of 8, then cap by memory
    with_buffer.max(8).min(memory_cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id() {
        let id = NodeId::new(42);
        assert_eq!(id.value(), 42);
        assert_eq!(format!("{}", id), "Node(42)");

        let id2: NodeId = 42.into();
        assert_eq!(id, id2);
    }

    #[test]
    fn test_pool_metrics() {
        let mut metrics = PoolMetrics::new();
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.hit_rate(), 0.0);

        metrics.hits = 80;
        metrics.misses = 20;
        assert!((metrics.hit_rate() - 80.0).abs() < 0.01);

        metrics.evictions = 5;
        metrics.peak_usage = 32;
        let display = format!("{}", metrics);
        assert!(display.contains("80.0%"));
    }

    #[test]
    fn test_recommended_pool_capacity() {
        // Depth-wise: depth 4 needs up to 2^5 = 32 nodes
        let cap = recommended_pool_capacity(31, Some(4), 1000, 1_000_000);
        assert!(cap >= 32); // At least 2^5
        assert!(cap <= 50); // Not too much more with buffer

        // Leaf-wise: 31 leaves × 2 = 62, + 25% = ~78
        let cap = recommended_pool_capacity(31, None, 1000, 1_000_000);
        assert!(cap >= 62);
        assert!(cap <= 100);

        // Memory cap kicks in
        let cap = recommended_pool_capacity(100, None, 10_000, 50_000);
        assert_eq!(cap, 5); // 50_000 / 10_000 = 5, but minimum is 8
        // Actually minimum is 8, so:
        let cap = recommended_pool_capacity(100, None, 10_000, 100_000);
        assert!(cap <= 10); // Memory cap should limit

        // Minimum 8
        let cap = recommended_pool_capacity(2, Some(1), 1000, 1_000_000);
        assert!(cap >= 8);
    }
}
