//! LRU histogram pool for gradient boosting tree training.
//!
//! This module provides a fixed-size cache for histogram arrays that maps logical
//! node IDs to physical storage slots using an LRU eviction policy.

use crate::repr::gbdt::NodeId;

use super::ops::{HistogramBin, clear_histogram, subtract_histogram};

/// Physical slot index in the histogram cache.
pub type SlotId = u32;

/// Sentinel value indicating an unmapped node or slot.
const UNMAPPED: i32 = -1;

/// Metadata describing a single feature's histogram layout.
#[derive(Clone, Debug)]
pub struct HistogramLayout {
    /// Bin offset into the histogram data buffer.
    pub offset: u32,
    /// Number of bins for this feature.
    pub n_bins: u32,
}

/// Result of acquiring a histogram slot for a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquireResult {
    /// Cache hit: the slot already contained valid data for this node.
    Hit(SlotId),
    /// Cache miss: a slot was allocated (possibly evicting another node's data).
    /// The caller must rebuild the histogram.
    Miss(SlotId),
}

impl AcquireResult {
    /// Returns the slot ID regardless of hit/miss.
    #[inline]
    pub fn slot(&self) -> SlotId {
        match *self {
            AcquireResult::Hit(s) | AcquireResult::Miss(s) => s,
        }
    }

    /// Returns true if this was a cache hit.
    #[inline]
    pub fn is_hit(&self) -> bool {
        matches!(self, AcquireResult::Hit(_))
    }
}

/// A fixed-size LRU cache for histogram arrays, mapping logical node IDs to physical slots.
///
/// # Design
///
/// During tree training, we need histogram arrays for active leaves. The number of
/// logical nodes can exceed the cache size, so we use an LRU policy to evict
/// least-recently-used histograms when the cache is full.
///
/// When `cache_size >= total_nodes`, the pool operates in "identity mode" where
/// `node_id == slot_id` and no LRU tracking is needed.
///
/// # Memory Layout
///
/// Histograms use row-major layout where each bin stores `(f64, f64)` tuples
/// for gradient and hessian sums.
///
/// ```text
/// Slot 0: [(grad0, hess0), (grad1, hess1), ..., (gradN, hessN)]
/// Slot 1: [(grad0, hess0), (grad1, hess1), ..., (gradN, hessN)]
/// ...
/// ```
///
/// # LRU Implementation
///
/// Uses timestamp-based LRU rather than a linked list:
/// - Each slot has a `last_used_time` counter
/// - On access, we increment a global `current_time` and store it
/// - On eviction, we find the slot with the minimum timestamp
///
/// This is O(cache_size) for eviction but simpler and has better cache locality
/// than a linked list. Since cache sizes are typically small (e.g., n_leaves),
/// this is efficient in practice.
pub struct HistogramPool {
    /// Per-slot histogram data. Each slot stores `total_bins` bins.
    /// Layout: [slot0_bin0, slot0_bin1, ..., slot1_bin0, ...]
    data: Box<[HistogramBin]>,

    /// Feature metadata (shared across all slots).
    features: Box<[HistogramLayout]>,

    /// Total number of bins per histogram (sum of all feature bins).
    total_bins: usize,

    /// Number of physical slots in the cache.
    cache_size: usize,

    /// Maximum number of logical nodes that may be mapped.
    #[allow(dead_code)]
    total_nodes: usize,

    /// Maps `node_id -> slot_id` (or UNMAPPED if not cached).
    /// Only used when `cache_size < total_nodes`.
    node_to_slot: Box<[i32]>,

    /// Maps `slot_id -> node_id` (or UNMAPPED if slot is free).
    /// Only used when `cache_size < total_nodes`.
    slot_to_node: Box<[i32]>,

    /// Timestamp of last access for each slot (for LRU eviction).
    /// Only used when `cache_size < total_nodes`.
    last_used_time: Box<[u64]>,

    /// Monotonically increasing counter for LRU timestamps.
    current_time: u64,

    /// When true, `cache_size >= total_nodes` so we use identity mapping.
    identity_mode: bool,

    /// Current epoch for identity-mode validity tracking.
    ///
    /// In identity mode, `node_id == slot_id` so we don't use mappings.
    /// We still must invalidate histograms between trees and when simulating
    /// `move_mapping` during the subtraction trick.
    identity_epoch: u64,

    /// Per-slot epoch marker for identity mode.
    ///
    /// A slot is considered valid iff `identity_slot_epoch[slot] == identity_epoch`.
    identity_slot_epoch: Box<[u64]>,

    /// Eviction pins (by slot). Only used when `!identity_mode`.
    pinned: Box<[bool]>,
}

impl HistogramPool {
    /// Creates a new histogram pool.
    ///
    /// # Arguments
    ///
    /// * `features` - Metadata for each feature (bin counts, offsets)
    /// * `cache_size` - Number of physical histogram slots to allocate
    /// * `total_nodes` - Maximum number of logical nodes that may be mapped
    ///
    /// # Panics
    ///
    /// Panics if `cache_size < 2` (need at least slots for smaller and larger leaf).
    pub fn new(features: Vec<HistogramLayout>, cache_size: usize, total_nodes: usize) -> Self {
        assert!(cache_size >= 2, "cache_size must be at least 2");

        let cache_size = cache_size.min(total_nodes);
        let identity_mode = cache_size >= total_nodes;

        // Calculate total bins from feature metadata
        let total_bins: usize = features.iter().map(|f| f.n_bins as usize).sum();

        // Allocate data: one bin per slot
        let data_len = cache_size * total_bins;
        let data = vec![(0.0, 0.0); data_len].into_boxed_slice();

        // Allocate mapping arrays only if needed
        let (node_to_slot, slot_to_node, last_used_time, pinned) = if identity_mode {
            (
                Box::default(),
                Box::default(),
                Box::default(),
                Box::default(),
            )
        } else {
            (
                vec![UNMAPPED; total_nodes].into_boxed_slice(),
                vec![UNMAPPED; cache_size].into_boxed_slice(),
                vec![0u64; cache_size].into_boxed_slice(),
                vec![false; cache_size].into_boxed_slice(),
            )
        };

        let identity_slot_epoch = if identity_mode {
            vec![0u64; cache_size].into_boxed_slice()
        } else {
            Box::default()
        };

        Self {
            data,
            features: features.into_boxed_slice(),
            total_bins,
            cache_size,
            total_nodes,
            node_to_slot,
            slot_to_node,
            last_used_time,
            current_time: 0,
            identity_mode,

            identity_epoch: 1,
            identity_slot_epoch,
            pinned,
        }
    }

    /// Resets all mappings, invalidating cached histograms.
    ///
    /// Call this at the start of each new tree.
    pub fn reset_mappings(&mut self) {
        if self.identity_mode {
            // In identity mode, we don't use mappings. Instead, bump the epoch so that
            // subsequent `acquire(node)` returns Miss and forces a rebuild/clear.
            self.identity_epoch = self.identity_epoch.wrapping_add(1);
            if self.identity_epoch == 0 {
                // Avoid ambiguous "never valid" epoch.
                self.identity_epoch = 1;
            }
            return;
        }

        self.current_time = 0;
        self.node_to_slot.fill(UNMAPPED);
        self.slot_to_node.fill(UNMAPPED);
        self.last_used_time.fill(0);
        self.pinned.fill(false);
    }

    /// Pin a node's histogram slot (best-effort).
    ///
    /// When pinned, the slot will not be chosen for LRU eviction.
    pub fn pin(&mut self, node_id: NodeId) {
        if self.identity_mode {
            return;
        }
        if let Some(slot) = self.resolve_slot(node_id) {
            self.pinned[slot as usize] = true;
        }
    }

    /// Unpin a node's histogram slot (best-effort).
    pub fn unpin(&mut self, node_id: NodeId) {
        if self.identity_mode {
            return;
        }
        if let Some(slot) = self.resolve_slot(node_id) {
            self.pinned[slot as usize] = false;
        }
    }

    /// Acquires a histogram slot for the given node.
    ///
    /// Returns `Hit` if the node already has a cached histogram, or `Miss` if
    /// a new slot was allocated (possibly evicting another node).
    ///
    /// On a miss, the histogram data is stale and must be rebuilt by the caller.
    pub fn acquire(&mut self, node_id: NodeId) -> AcquireResult {
        let node_idx = node_id as usize;

        if self.identity_mode {
            // Identity mode: node_id == slot_id.
            // We still need per-tree validity; treat as miss if this slot hasn't
            // been built in the current epoch.
            if node_idx >= self.cache_size {
                return AcquireResult::Miss(node_id);
            }

            let slot_epoch = self.identity_slot_epoch[node_idx];
            if slot_epoch == self.identity_epoch {
                return AcquireResult::Hit(node_id);
            }

            self.identity_slot_epoch[node_idx] = self.identity_epoch;
            return AcquireResult::Miss(node_id);
        }

        // Check if node is already mapped
        let existing_slot = self.node_to_slot[node_idx];
        if existing_slot >= 0 {
            let slot = existing_slot as SlotId;
            self.current_time += 1;
            self.last_used_time[slot as usize] = self.current_time;
            return AcquireResult::Hit(slot);
        }

        // Cache miss: find LRU slot to evict
        let slot = self.find_lru_slot();
        self.current_time += 1;
        self.last_used_time[slot as usize] = self.current_time;

        // Evict previous occupant if any
        let prev_node = self.slot_to_node[slot as usize];
        if prev_node >= 0 {
            self.node_to_slot[prev_node as usize] = UNMAPPED;
        }

        // Map new node to slot
        self.node_to_slot[node_idx] = slot as i32;
        self.slot_to_node[slot as usize] = node_id as i32;

        AcquireResult::Miss(slot)
    }

    /// Releases a node's histogram slot, marking it as available for eviction.
    ///
    /// This is optional - the LRU policy will eventually evict unused slots.
    /// Use this when you know a node's histogram is no longer needed.
    pub fn release(&mut self, node_id: NodeId) {
        if self.identity_mode {
            return;
        }

        let node_idx = node_id as usize;
        let slot = self.node_to_slot[node_idx];
        if slot >= 0 {
            self.node_to_slot[node_idx] = UNMAPPED;
            self.slot_to_node[slot as usize] = UNMAPPED;
            self.last_used_time[slot as usize] = 0; // Make it most eligible for eviction
            self.pinned[slot as usize] = false;
        }
    }

    /// Moves a histogram from one node to another.
    ///
    /// This is used during tree growth when a parent's histogram is reused
    /// by a child via the histogram subtraction trick.
    ///
    /// After this call, `from_node` is unmapped and `to_node` owns the slot.
    pub fn move_mapping(&mut self, from_node: NodeId, to_node: NodeId) {
        if self.identity_mode {
            // In identity mode, we need to actually copy the underlying data
            let from_slot = from_node as usize;
            let to_slot = to_node as usize;

            // No-op if moving to self.
            if from_slot == to_slot {
                return;
            }

            let from_start = from_slot * self.total_bins;
            let to_start = to_slot * self.total_bins;

            // Copy the data between slots
            for i in 0..self.total_bins {
                self.data[to_start + i] = self.data[from_start + i];
            }

            // Simulate "moving" ownership semantics used by the subtraction trick:
            // - destination becomes valid in the current epoch
            // - source becomes invalid so a subsequent acquire(source) returns Miss
            if from_slot < self.cache_size && to_slot < self.cache_size {
                self.identity_slot_epoch[to_slot] = self.identity_epoch;
                // Invalidate the source so a subsequent acquire(from_node) rebuilds.
                self.identity_slot_epoch[from_slot] = 0;
            }
            return;
        }

        let from_idx = from_node as usize;
        let slot = self.node_to_slot[from_idx];
        if slot < 0 {
            return; // Source not mapped, nothing to do
        }

        // Unmap source
        self.node_to_slot[from_idx] = UNMAPPED;

        // Map destination
        let to_idx = to_node as usize;
        self.node_to_slot[to_idx] = slot;
        self.slot_to_node[slot as usize] = to_node as i32;

        // Update timestamp
        self.current_time += 1;
        self.last_used_time[slot as usize] = self.current_time;
    }

    /// Returns a read-only view of the histogram for a node, if cached.
    pub fn get(&self, node_id: NodeId) -> Option<HistogramSlot<'_>> {
        let slot = self.resolve_slot(node_id)?;
        Some(self.slot(slot))
    }

    /// Returns a mutable view of the histogram for a node, if cached.
    pub fn get_mut(&mut self, node_id: NodeId) -> Option<HistogramSlotMut<'_>> {
        let slot = self.resolve_slot(node_id)?;
        Some(self.slot_mut(slot))
    }

    /// Returns a read-only view of a histogram slot by slot ID.
    pub fn slot(&self, slot_id: SlotId) -> HistogramSlot<'_> {
        let slot = slot_id as usize;
        let slot_start = slot * self.total_bins;
        let slot_end = slot_start + self.total_bins;

        HistogramSlot {
            bins: &self.data[slot_start..slot_end],
            features: &self.features,
        }
    }

    /// Returns a mutable view of a histogram slot by slot ID.
    pub fn slot_mut(&mut self, slot_id: SlotId) -> HistogramSlotMut<'_> {
        let slot_start = slot_id as usize * self.total_bins;
        let slot_end = slot_start + self.total_bins;

        HistogramSlotMut {
            bins: &mut self.data[slot_start..slot_end],
            features: &self.features,
        }
    }

    /// Subtract source histogram from target histogram in-place.
    ///
    /// This is used for the subtraction trick during tree growth:
    /// `target = target - source` (target typically contains parent's histogram).
    ///
    /// # Panics
    /// Panics if source or target nodes are not mapped.
    pub fn subtract(&mut self, target: NodeId, source: NodeId) {
        let target_slot = self
            .resolve_slot(target)
            .expect("Target histogram not found");
        let source_slot = self
            .resolve_slot(source)
            .expect("Source histogram not found");

        let target_start = target_slot as usize * self.total_bins;
        let source_start = source_slot as usize * self.total_bins;

        let (target_slice, source_slice) = crate::utils::disjoint_slices_mut(
            &mut self.data,
            target_start,
            source_start,
            self.total_bins,
        );
        subtract_histogram(target_slice, source_slice);
    }

    /// Returns the feature metadata.
    #[inline]
    pub fn features(&self) -> &[HistogramLayout] {
        &self.features
    }

    /// Returns the total number of bins per histogram.
    #[inline]
    pub fn total_bins(&self) -> usize {
        self.total_bins
    }

    /// Returns the cache size (number of physical slots).
    #[inline]
    pub fn cache_size(&self) -> usize {
        self.cache_size
    }

    // --- Private helpers ---

    /// Resolves a node ID to its slot ID, if mapped.
    #[inline]
    fn resolve_slot(&self, node_id: NodeId) -> Option<SlotId> {
        if self.identity_mode {
            let slot = node_id as usize;
            if slot >= self.cache_size {
                return None;
            }

            // Only consider the slot mapped if it has been built in this epoch.
            if self.identity_slot_epoch[slot] == self.identity_epoch {
                Some(node_id)
            } else {
                None
            }
        } else {
            let slot = self.node_to_slot[node_id as usize];
            if slot >= 0 {
                Some(slot as SlotId)
            } else {
                None
            }
        }
    }

    /// Finds the slot with the smallest timestamp (least recently used).
    fn find_lru_slot(&self) -> SlotId {
        let mut min_time = u64::MAX;
        let mut lru_slot = 0;
        let mut found = false;

        for (slot, &time) in self.last_used_time.iter().enumerate() {
            if self.pinned[slot] {
                continue;
            }
            if time < min_time {
                min_time = time;
                lru_slot = slot;
                found = true;
            }
        }

        assert!(found, "All histogram slots are pinned; cannot evict");
        lru_slot as SlotId
    }
}

/// Read-only view of a single histogram slot.
pub struct HistogramSlot<'a> {
    /// Histogram bins (grad, hess) tuples.
    pub bins: &'a [HistogramBin],
    /// Feature metadata for iterating by feature.
    pub features: &'a [HistogramLayout],
}

impl<'a> HistogramSlot<'a> {
    /// Returns bins for a specific feature.
    #[inline]
    pub fn feature_bins(&self, feature_idx: usize) -> &[HistogramBin] {
        let meta = &self.features[feature_idx];
        let start = meta.offset as usize;
        let end = start + meta.n_bins as usize;
        &self.bins[start..end]
    }

    /// Iterates over a feature's bins, yielding (grad_sum, hess_sum) pairs.
    pub fn iter_feature(&self, feature_idx: usize) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.feature_bins(feature_idx).iter().copied()
    }

    /// Returns the number of features.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.features.len()
    }

    /// Sum all histogram bins to get total gradient and hessian sums.
    ///
    /// This is O(n_bins) instead of O(n_rows), which is much faster for large leaves.
    /// Each row contributes to exactly one bin per feature, so summing one feature's
    /// bins gives the total gradient/hessian sums for all rows in this histogram.
    #[inline]
    pub fn sum_gradients(&self) -> (f64, f64) {
        // Sum the first feature's bins (all features contain the same rows)
        if self.features.is_empty() {
            return (0.0, 0.0);
        }
        let bins = self.feature_bins(0);
        let mut sum_g = 0.0f64;
        let mut sum_h = 0.0f64;
        for &(g, h) in bins {
            sum_g += g;
            sum_h += h;
        }
        (sum_g, sum_h)
    }
}

/// Mutable view of a single histogram slot.
pub struct HistogramSlotMut<'a> {
    /// Histogram bins (mutable).
    pub bins: &'a mut [HistogramBin],
    /// Feature metadata for iterating by feature.
    pub features: &'a [HistogramLayout],
}

impl<'a> HistogramSlotMut<'a> {
    /// Returns bins for a specific feature.
    #[inline]
    pub fn feature_bins(&self, feature_idx: usize) -> &[HistogramBin] {
        let meta = &self.features[feature_idx];
        let start = meta.offset as usize;
        let end = start + meta.n_bins as usize;
        &self.bins[start..end]
    }

    /// Returns mutable bins for a specific feature.
    #[inline]
    pub fn feature_bins_mut(&mut self, feature_idx: usize) -> &mut [HistogramBin] {
        let meta = &self.features[feature_idx];
        let start = meta.offset as usize;
        let end = start + meta.n_bins as usize;
        &mut self.bins[start..end]
    }

    /// Iterates over a feature's bins, yielding (grad_sum, hess_sum) pairs.
    pub fn iter_feature(&self, feature_idx: usize) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.feature_bins(feature_idx).iter().copied()
    }

    /// Clears the histogram by resetting all bins to zero.
    pub fn clear(&mut self) {
        clear_histogram(self.bins);
    }

    /// Returns the number of features.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.features.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features(bin_counts: &[u32]) -> Vec<HistogramLayout> {
        let mut offset = 0;
        bin_counts
            .iter()
            .map(|&n_bins| {
                let meta = HistogramLayout { offset, n_bins };
                offset += n_bins;
                meta
            })
            .collect()
    }

    #[test]
    fn test_identity_mode() {
        let features = make_features(&[4, 8, 16]);
        let mut pool = HistogramPool::new(features, 10, 10);

        // In identity mode, node_id == slot_id, but validity is tracked per-epoch.
        // The first acquire in an epoch is a Miss and forces a rebuild/clear.
        assert!(pool.identity_mode);

        let result = pool.acquire(5);
        assert!(!result.is_hit());
        assert_eq!(result.slot(), 5);

        // Second acquire in the same epoch is a hit.
        let result2 = pool.acquire(5);
        assert!(result2.is_hit());

        // Can access the slot once acquired in the current epoch.
        assert!(pool.get(5).is_some());
        assert!(pool.get(11).is_none()); // Out of range

        // After reset (new epoch), must rebuild again.
        pool.reset_mappings();
        let result3 = pool.acquire(5);
        assert!(!result3.is_hit());
    }

    #[test]
    fn test_identity_mode_move_mapping_invalidates_source() {
        let features = make_features(&[4]);
        // Identity mode: cache_size == total_nodes
        let mut pool = HistogramPool::new(features, 4, 4);
        assert!(pool.identity_mode);

        // Acquire/build node 0 in current epoch.
        let slot0 = pool.acquire(0).slot();
        pool.slot_mut(slot0).bins[0] = (42.0, 10.0);

        // Copy/move to node 1 (subtraction trick uses this pattern).
        pool.move_mapping(0, 1);

        // Destination should be visible and contain copied data.
        let view1 = pool
            .get(1)
            .expect("node 1 should be valid after move_mapping");
        assert_eq!(view1.bins[0].0, 42.0);
        assert_eq!(view1.bins[0].1, 10.0);

        // Source should be invalidated (so rebuild clears old parent histogram).
        assert!(pool.get(0).is_none());
        let r = pool.acquire(0);
        assert!(!r.is_hit(), "expected Miss after source invalidation");
    }

    #[test]
    fn test_identity_mode_move_mapping_noop_self() {
        let features = make_features(&[4]);
        let mut pool = HistogramPool::new(features, 2, 2);
        assert!(pool.identity_mode);

        let slot0 = pool.acquire(0).slot();
        pool.slot_mut(slot0).bins[0] = (7.0, 3.0);

        // Moving to self should not invalidate or change data.
        pool.move_mapping(0, 0);
        let view0 = pool.get(0).expect("node 0 should remain valid");
        assert_eq!(view0.bins[0].0, 7.0);
        assert_eq!(view0.bins[0].1, 3.0);
        assert!(pool.acquire(0).is_hit());
    }

    #[test]
    fn test_lru_eviction() {
        let features = make_features(&[4, 4]);
        // Cache size 3, but up to 10 logical nodes
        let mut pool = HistogramPool::new(features, 3, 10);

        assert!(!pool.identity_mode);

        // Acquire nodes 0, 1, 2 - all misses
        assert!(!pool.acquire(0).is_hit());
        assert!(!pool.acquire(1).is_hit());
        assert!(!pool.acquire(2).is_hit());

        // Re-acquire node 1 - should be a hit
        assert!(pool.acquire(1).is_hit());

        // Acquire node 3 - miss, should evict node 0 (LRU)
        let result = pool.acquire(3);
        assert!(!result.is_hit());

        // Node 0 should be evicted
        assert!(pool.get(0).is_none());
        // Nodes 1, 2, 3 should still be cached
        assert!(pool.get(1).is_some());
        assert!(pool.get(2).is_some());
        assert!(pool.get(3).is_some());
    }

    #[test]
    fn test_move_mapping() {
        let features = make_features(&[4]);
        let mut pool = HistogramPool::new(features, 5, 10);

        // Acquire node 0
        let result = pool.acquire(0);
        let slot = result.slot();

        // Write some data
        pool.slot_mut(slot).bins[0] = (42.0, 10.0);

        // Move from node 0 to node 5
        pool.move_mapping(0, 5);

        // Node 0 should be unmapped
        assert!(pool.get(0).is_none());

        // Node 5 should have the data
        let view = pool.get(5).unwrap();
        assert_eq!(view.bins[0].0, 42.0);
        assert_eq!(view.bins[0].1, 10.0);
    }

    #[test]
    fn test_reset_mappings() {
        let features = make_features(&[4]);
        let mut pool = HistogramPool::new(features, 3, 10);

        pool.acquire(0);
        pool.acquire(1);
        pool.acquire(2);

        assert!(pool.get(0).is_some());
        assert!(pool.get(1).is_some());

        pool.reset_mappings();

        // All mappings should be cleared
        assert!(pool.get(0).is_none());
        assert!(pool.get(1).is_none());
        assert!(pool.get(2).is_none());
    }

    #[test]
    fn test_feature_accessors() {
        let features = make_features(&[4, 8, 2]);
        let mut pool = HistogramPool::new(features, 2, 2);

        let result = pool.acquire(0);
        let slot = result.slot();

        {
            let mut view = pool.slot_mut(slot);

            // Write to feature 1's bins
            let bins = view.feature_bins_mut(1);
            assert_eq!(bins.len(), 8);
            bins[0] = (1.0, 0.5);
            bins[7] = (8.0, 4.0);
        }

        // Verify the data
        let view = pool.slot(slot);
        let bins = view.feature_bins(1);
        assert_eq!(bins[0].0, 1.0);
        assert_eq!(bins[0].1, 0.5);
        assert_eq!(bins[7].0, 8.0);
        assert_eq!(bins[7].1, 4.0);
    }

    #[test]
    fn test_iter_feature() {
        let features = make_features(&[3]);
        let mut pool = HistogramPool::new(features, 2, 2);

        let slot = pool.acquire(0).slot();
        {
            let view = pool.slot_mut(slot);
            view.bins[0] = (1.0, 0.1);
            view.bins[1] = (2.0, 0.2);
            view.bins[2] = (3.0, 0.3);
        }

        let view = pool.slot(slot);
        let sums: Vec<_> = view.iter_feature(0).collect();
        assert_eq!(sums, vec![(1.0, 0.1), (2.0, 0.2), (3.0, 0.3)]);
    }

    #[test]
    fn test_clear() {
        let features = make_features(&[4]);
        let mut pool = HistogramPool::new(features, 2, 2);

        let slot = pool.acquire(0).slot();
        {
            let mut view = pool.slot_mut(slot);
            view.bins[0] = (42.0, 10.0);
            view.clear();
        }

        let view = pool.slot(slot);
        assert_eq!(view.bins[0].0, 0.0);
        assert_eq!(view.bins[0].1, 0.0);
    }

    #[test]
    fn test_pinning_prevents_eviction() {
        let features = vec![HistogramLayout {
            offset: 0,
            n_bins: 4,
        }];
        let mut pool = HistogramPool::new(features, 2, 10);

        pool.acquire(0);
        pool.acquire(1);
        assert!(pool.get(0).is_some());
        assert!(pool.get(1).is_some());

        pool.pin(0);
        pool.acquire(2);

        assert!(pool.get(0).is_some());
        assert!(pool.get(2).is_some());
        assert!(pool.get(1).is_none());
    }
}
