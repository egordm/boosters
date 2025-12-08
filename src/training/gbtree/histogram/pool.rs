//! Contiguous histogram pool with LRU eviction.
//!
//! This module provides memory-efficient histogram storage for tree building.
//! Histograms are stored in a single contiguous allocation with LRU eviction
//! when the pool is full.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                  ContiguousHistogramPool                         │
//! │  ┌───────────────────────────────────────────────────────────┐  │
//! │  │  Backing Store (SoA layout):                               │  │
//! │  │  sum_grads: [slot0_bins][slot1_bins][slot2_bins]...        │  │
//! │  │  sum_hess:  [slot0_bins][slot1_bins][slot2_bins]...        │  │
//! │  │  counts:    [slot0_bins][slot1_bins][slot2_bins]...        │  │
//! │  └───────────────────────────────────────────────────────────┘  │
//! │                                                                  │
//! │  node_to_slot: HashMap<NodeId, SlotId>  (O(1) lookup)           │
//! │  slot_to_node: Vec<Option<NodeId>>      (for eviction)          │
//! │  free_slots:   Vec<SlotId>              (LIFO for warmth)       │
//! │  lru_order:    VecDeque<SlotId>         (LRU tracking)          │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! See RFC-0025 for design rationale.

use std::collections::{HashMap, VecDeque};

use super::types::{NodeId, PoolMetrics, SlotId};

/// View into a single histogram slot (immutable).
///
/// Provides access to the three SoA arrays for one histogram.
#[derive(Debug)]
pub struct HistogramSlot<'a> {
    /// Sum of gradients per bin.
    pub sum_grad: &'a [f32],
    /// Sum of hessians per bin.
    pub sum_hess: &'a [f32],
    /// Count per bin.
    pub count: &'a [u32],
}

/// View into a single histogram slot (mutable).
///
/// Provides mutable access to the three SoA arrays for one histogram.
#[derive(Debug)]
pub struct HistogramSlotMut<'a> {
    /// Sum of gradients per bin.
    pub sum_grad: &'a mut [f32],
    /// Sum of hessians per bin.
    pub sum_hess: &'a mut [f32],
    /// Count per bin.
    pub count: &'a mut [u32],
}

impl HistogramSlotMut<'_> {
    /// Reset all bins to zero.
    #[inline]
    pub fn reset(&mut self) {
        self.sum_grad.fill(0.0);
        self.sum_hess.fill(0.0);
        self.count.fill(0);
    }

    /// Number of bins in this histogram.
    #[inline]
    pub fn num_bins(&self) -> usize {
        self.sum_grad.len()
    }

    /// Add a sample to a bin.
    #[inline]
    pub fn add(&mut self, bin: usize, grad: f32, hess: f32) {
        debug_assert!(bin < self.sum_grad.len());
        unsafe {
            *self.sum_grad.get_unchecked_mut(bin) += grad;
            *self.sum_hess.get_unchecked_mut(bin) += hess;
            *self.count.get_unchecked_mut(bin) += 1;
        }
    }
}

impl HistogramSlot<'_> {
    /// Number of bins in this histogram.
    #[inline]
    pub fn num_bins(&self) -> usize {
        self.sum_grad.len()
    }

    /// Get statistics for a bin.
    #[inline]
    pub fn bin_stats(&self, bin: usize) -> (f32, f32, u32) {
        debug_assert!(bin < self.sum_grad.len());
        unsafe {
            (
                *self.sum_grad.get_unchecked(bin),
                *self.sum_hess.get_unchecked(bin),
                *self.count.get_unchecked(bin),
            )
        }
    }

    /// Compute total gradient sum across all bins.
    pub fn total_grad(&self) -> f32 {
        self.sum_grad.iter().sum()
    }

    /// Compute total hessian sum across all bins.
    pub fn total_hess(&self) -> f32 {
        self.sum_hess.iter().sum()
    }

    /// Compute total count across all bins.
    ///
    /// Note: For flat histograms with multiple features, this returns the
    /// sum across ALL bins. Use `total_count_first_feature` if you need
    /// only the row count.
    pub fn total_count(&self) -> u32 {
        self.count.iter().sum()
    }

    /// Compute total count using only the first feature's bins.
    ///
    /// This matches the semantics of `NodeHistogram::total_count()` which
    /// only uses feature 0 to compute the row count. This is the correct
    /// method to call when you have a flat histogram with multiple features
    /// and want the actual number of rows.
    pub fn total_count_first_feature(&self, bins_for_first_feature: usize) -> u32 {
        self.count[..bins_for_first_feature].iter().sum()
    }

    /// Compute total gradient using only the first feature's bins.
    ///
    /// This matches the semantics of `NodeHistogram::total_grad()` which
    /// only uses feature 0 to compute the total gradient.
    pub fn total_grad_first_feature(&self, bins_for_first_feature: usize) -> f32 {
        self.sum_grad[..bins_for_first_feature].iter().sum()
    }

    /// Compute total hessian using only the first feature's bins.
    ///
    /// This matches the semantics of `NodeHistogram::total_hess()` which
    /// only uses feature 0 to compute the total hessian.
    pub fn total_hess_first_feature(&self, bins_for_first_feature: usize) -> f32 {
        self.sum_hess[..bins_for_first_feature].iter().sum()
    }
}

/// Pool of pre-allocated histograms in contiguous memory.
///
/// Uses Structure-of-Arrays (SoA) layout matching our existing `FeatureHistogram`:
/// separate contiguous arrays for gradients, hessians, and counts.
///
/// # Features
///
/// - **Single allocation**: All histograms in one contiguous buffer (cache-friendly)
/// - **LRU eviction**: When pool is full, evicts least recently used histogram
/// - **O(1) access**: HashMap lookup by node ID
/// - **Metrics tracking**: Hits, misses, evictions for tuning
///
/// # Memory Usage
///
/// Total memory: `capacity × bins_per_hist × 12` bytes
/// (4 bytes each for grad, hess, and count per bin)
///
/// # Example
///
/// ```ignore
/// let mut pool = ContiguousHistogramPool::new(32, 25600); // 32 slots, 100 features × 256 bins
///
/// // Get or allocate histogram for node 0
/// let mut hist = pool.get_or_allocate(NodeId(0));
/// hist.reset();
/// // ... build histogram ...
///
/// // Later: retrieve it
/// if let Some(hist) = pool.get(NodeId(0)) {
///     // Use cached histogram
/// }
///
/// // Explicitly release when done
/// pool.release(NodeId(0));
/// ```
pub struct ContiguousHistogramPool {
    /// Contiguous backing store for gradient sums.
    /// Layout: [slot0_bins][slot1_bins][slot2_bins]...
    sum_grads: Box<[f32]>,

    /// Contiguous backing store for hessian sums.
    sum_hess: Box<[f32]>,

    /// Contiguous backing store for sample counts.
    counts: Box<[u32]>,

    /// Number of bins per histogram (same for all).
    bins_per_hist: usize,

    /// Total number of histogram slots.
    capacity: usize,

    /// Maps node ID to slot index.
    node_to_slot: HashMap<NodeId, SlotId>,

    /// Maps slot index back to node ID (for eviction).
    slot_to_node: Vec<Option<NodeId>>,

    /// Free slots (LIFO for cache warmth).
    free_slots: Vec<SlotId>,

    /// LRU tracking: most recently used at back.
    /// Only contains slots that are in use.
    lru_order: VecDeque<SlotId>,

    /// Position in LRU queue for O(1) removal.
    /// `slot_to_lru_pos[slot.index()]` = position in `lru_order`, or None if not in LRU.
    slot_to_lru_pos: Vec<Option<usize>>,

    /// Usage statistics.
    metrics: PoolMetrics,
}

impl ContiguousHistogramPool {
    /// Create a new pool with the given capacity and histogram size.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of histograms the pool can hold
    /// * `bins_per_hist` - Total bins per histogram (sum across all features)
    ///
    /// # Memory
    ///
    /// Allocates `capacity × bins_per_hist × 12` bytes upfront.
    pub fn new(capacity: usize, bins_per_hist: usize) -> Self {
        let total_bins = capacity * bins_per_hist;

        // Allocate backing stores
        let sum_grads = vec![0.0f32; total_bins].into_boxed_slice();
        let sum_hess = vec![0.0f32; total_bins].into_boxed_slice();
        let counts = vec![0u32; total_bins].into_boxed_slice();

        // All slots start free (LIFO order: last slot first for cache warmth)
        let free_slots: Vec<SlotId> = (0..capacity as u32).rev().map(SlotId).collect();

        Self {
            sum_grads,
            sum_hess,
            counts,
            bins_per_hist,
            capacity,
            node_to_slot: HashMap::with_capacity(capacity),
            slot_to_node: vec![None; capacity],
            free_slots,
            lru_order: VecDeque::with_capacity(capacity),
            slot_to_lru_pos: vec![None; capacity],
            metrics: PoolMetrics::default(),
        }
    }

    /// Get or allocate histogram for a node.
    ///
    /// - If node already has a histogram, returns it (cache hit)
    /// - If pool has free slots, allocates one
    /// - If pool is full, evicts LRU node and reuses its slot
    ///
    /// The returned histogram is NOT reset - call `reset()` if needed.
    ///
    /// # Panics
    ///
    /// Panics if capacity is 0.
    pub fn get_or_allocate(&mut self, node: NodeId) -> HistogramSlotMut<'_> {
        // Check if already in pool
        if let Some(&slot) = self.node_to_slot.get(&node) {
            self.metrics.hits += 1;
            self.touch_lru(slot);
            return self.slot_mut(slot);
        }

        // Miss - need to allocate
        self.metrics.misses += 1;

        let slot = if let Some(slot) = self.free_slots.pop() {
            // Use a free slot
            slot
        } else {
            // Pool full - evict LRU
            self.evict_lru()
        };

        // Map node to slot
        self.node_to_slot.insert(node, slot);
        self.slot_to_node[slot.index()] = Some(node);

        // Add to LRU (most recently used)
        self.add_to_lru(slot);

        // Update peak usage
        let in_use = self.in_use();
        if in_use > self.metrics.peak_usage {
            self.metrics.peak_usage = in_use;
        }

        self.slot_mut(slot)
    }

    /// Get existing histogram for node (if present).
    ///
    /// Does NOT allocate. Returns `None` if node not in pool.
    /// Updates LRU position (marks as recently used).
    pub fn get(&mut self, node: NodeId) -> Option<HistogramSlot<'_>> {
        let slot = *self.node_to_slot.get(&node)?;
        self.metrics.hits += 1;
        self.touch_lru(slot);
        Some(self.slot(slot))
    }

    /// Get mutable reference without LRU update.
    ///
    /// Used during histogram building when slot is already "in use"
    /// and we don't want to update LRU position repeatedly.
    pub fn get_mut_no_lru(&mut self, node: NodeId) -> Option<HistogramSlotMut<'_>> {
        let slot = *self.node_to_slot.get(&node)?;
        Some(self.slot_mut(slot))
    }

    /// Check if node has histogram in pool.
    #[inline]
    pub fn contains(&self, node: NodeId) -> bool {
        self.node_to_slot.contains_key(&node)
    }

    /// Explicitly release a node's histogram.
    ///
    /// Use when a node is fully expanded and its histogram is definitely not needed.
    /// More efficient than waiting for LRU eviction.
    pub fn release(&mut self, node: NodeId) {
        if let Some(slot) = self.node_to_slot.remove(&node) {
            self.slot_to_node[slot.index()] = None;
            self.remove_from_lru(slot);
            self.free_slots.push(slot);
        }
    }

    /// Release multiple nodes at once.
    ///
    /// More efficient than individual releases when many nodes are done
    /// (e.g., when a tree level is complete).
    pub fn release_batch(&mut self, nodes: &[NodeId]) {
        for &node in nodes {
            self.release(node);
        }
    }

    /// Reset pool for new tree (keeps allocations).
    ///
    /// Clears all node mappings but retains the allocated memory.
    /// Call this before building a new tree.
    pub fn reset(&mut self) {
        self.node_to_slot.clear();
        self.slot_to_node.fill(None);
        self.free_slots.clear();
        self.free_slots
            .extend((0..self.capacity as u32).rev().map(SlotId));
        self.lru_order.clear();
        self.slot_to_lru_pos.fill(None);
        // Note: metrics are NOT reset - call metrics.reset() separately if desired
    }

    /// Get pool statistics.
    #[inline]
    pub fn metrics(&self) -> &PoolMetrics {
        &self.metrics
    }

    /// Get mutable reference to metrics (for resetting).
    #[inline]
    pub fn metrics_mut(&mut self) -> &mut PoolMetrics {
        &mut self.metrics
    }

    /// Number of histograms currently in use.
    #[inline]
    pub fn in_use(&self) -> usize {
        self.capacity - self.free_slots.len()
    }

    /// Number of free slots available.
    #[inline]
    pub fn available(&self) -> usize {
        self.free_slots.len()
    }

    /// Total capacity of the pool.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of bins per histogram.
    #[inline]
    pub fn bins_per_hist(&self) -> usize {
        self.bins_per_hist
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    /// Get immutable view of a slot's histogram.
    fn slot(&self, slot: SlotId) -> HistogramSlot<'_> {
        let start = slot.index() * self.bins_per_hist;
        let end = start + self.bins_per_hist;
        HistogramSlot {
            sum_grad: &self.sum_grads[start..end],
            sum_hess: &self.sum_hess[start..end],
            count: &self.counts[start..end],
        }
    }

    /// Get mutable view of a slot's histogram.
    fn slot_mut(&mut self, slot: SlotId) -> HistogramSlotMut<'_> {
        let start = slot.index() * self.bins_per_hist;
        let end = start + self.bins_per_hist;
        HistogramSlotMut {
            sum_grad: &mut self.sum_grads[start..end],
            sum_hess: &mut self.sum_hess[start..end],
            count: &mut self.counts[start..end],
        }
    }

    /// Add slot to LRU queue (most recently used position).
    fn add_to_lru(&mut self, slot: SlotId) {
        let pos = self.lru_order.len();
        self.slot_to_lru_pos[slot.index()] = Some(pos);
        self.lru_order.push_back(slot);
    }

    /// Update LRU position when slot is accessed.
    fn touch_lru(&mut self, slot: SlotId) {
        // Remove from current position
        if let Some(pos) = self.slot_to_lru_pos[slot.index()] {
            self.lru_order.remove(pos);

            // Update positions for slots that shifted
            for (i, &s) in self.lru_order.iter().enumerate().skip(pos) {
                self.slot_to_lru_pos[s.index()] = Some(i);
            }
        }

        // Add to back (most recent)
        let new_pos = self.lru_order.len();
        self.slot_to_lru_pos[slot.index()] = Some(new_pos);
        self.lru_order.push_back(slot);
    }

    /// Remove slot from LRU queue.
    fn remove_from_lru(&mut self, slot: SlotId) {
        if let Some(pos) = self.slot_to_lru_pos[slot.index()].take() {
            self.lru_order.remove(pos);

            // Update positions for slots that shifted
            for (i, &s) in self.lru_order.iter().enumerate().skip(pos) {
                self.slot_to_lru_pos[s.index()] = Some(i);
            }
        }
    }

    /// Evict least recently used slot and return it for reuse.
    fn evict_lru(&mut self) -> SlotId {
        let slot = self
            .lru_order
            .pop_front()
            .expect("LRU empty but pool full");

        // Update LRU positions after removal
        for (i, &s) in self.lru_order.iter().enumerate() {
            self.slot_to_lru_pos[s.index()] = Some(i);
        }

        // Remove node mapping
        if let Some(node) = self.slot_to_node[slot.index()].take() {
            self.node_to_slot.remove(&node);
        }

        self.slot_to_lru_pos[slot.index()] = None;
        self.metrics.evictions += 1;

        slot
    }
}

impl std::fmt::Debug for ContiguousHistogramPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContiguousHistogramPool")
            .field("capacity", &self.capacity)
            .field("bins_per_hist", &self.bins_per_hist)
            .field("in_use", &self.in_use())
            .field("metrics", &self.metrics)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_basic_allocation() {
        let mut pool = ContiguousHistogramPool::new(4, 10);

        assert_eq!(pool.capacity(), 4);
        assert_eq!(pool.available(), 4);
        assert_eq!(pool.in_use(), 0);

        // Allocate histogram for node 0
        let hist = pool.get_or_allocate(NodeId(0));
        assert_eq!(hist.num_bins(), 10);

        assert_eq!(pool.in_use(), 1);
        assert_eq!(pool.available(), 3);
        assert!(pool.contains(NodeId(0)));

        // Stats should show a miss
        assert_eq!(pool.metrics().hits, 0);
        assert_eq!(pool.metrics().misses, 1);
    }

    #[test]
    fn test_pool_hit_on_second_access() {
        let mut pool = ContiguousHistogramPool::new(4, 10);

        // First access - miss
        let mut hist = pool.get_or_allocate(NodeId(0));
        hist.reset();
        hist.add(0, 1.0, 0.5);

        assert_eq!(pool.metrics().misses, 1);
        assert_eq!(pool.metrics().hits, 0);

        // Second access - hit
        let hist = pool.get(NodeId(0)).unwrap();
        assert_eq!(hist.bin_stats(0), (1.0, 0.5, 1));

        assert_eq!(pool.metrics().hits, 1);
        assert_eq!(pool.metrics().misses, 1);
    }

    #[test]
    fn test_pool_lru_eviction() {
        let mut pool = ContiguousHistogramPool::new(3, 10);

        // Fill pool
        for i in 0..3 {
            let mut hist = pool.get_or_allocate(NodeId(i));
            hist.reset();
            hist.add(0, i as f32, 1.0);
        }

        assert_eq!(pool.in_use(), 3);
        assert_eq!(pool.available(), 0);

        // Access node 0 to make it recently used
        let _ = pool.get(NodeId(0));

        // LRU order should now be: [1, 2, 0] (oldest to newest)

        // Allocate node 3 - should evict node 1 (LRU)
        let mut hist = pool.get_or_allocate(NodeId(3));
        hist.reset();

        assert!(pool.contains(NodeId(0))); // Still there (was accessed)
        assert!(!pool.contains(NodeId(1))); // Evicted (was LRU)
        assert!(pool.contains(NodeId(2)));
        assert!(pool.contains(NodeId(3)));

        assert_eq!(pool.metrics().evictions, 1);
    }

    #[test]
    fn test_pool_lru_order() {
        let mut pool = ContiguousHistogramPool::new(3, 10);

        // Allocate in order: 0, 1, 2
        for i in 0..3 {
            pool.get_or_allocate(NodeId(i));
        }

        // Access in order: 2, 0, 1
        // LRU order after: [2, 0, 1] → wait, that's not right
        // Let's trace:
        // Initial LRU (oldest to newest): [0, 1, 2]
        // Access 2: LRU becomes [0, 1, 2] → remove 2, add to back → [0, 1, 2]
        // Access 0: LRU becomes [1, 2, 0]
        // Access 1: LRU becomes [2, 0, 1]

        let _ = pool.get(NodeId(2));
        let _ = pool.get(NodeId(0));
        let _ = pool.get(NodeId(1));

        // Now allocate node 3 - should evict node 2 (LRU)
        pool.get_or_allocate(NodeId(3));

        assert!(!pool.contains(NodeId(2))); // Evicted
        assert!(pool.contains(NodeId(0)));
        assert!(pool.contains(NodeId(1)));
        assert!(pool.contains(NodeId(3)));
    }

    #[test]
    fn test_pool_release() {
        let mut pool = ContiguousHistogramPool::new(4, 10);

        // Allocate and release
        pool.get_or_allocate(NodeId(0));
        pool.get_or_allocate(NodeId(1));

        assert_eq!(pool.in_use(), 2);

        pool.release(NodeId(0));

        assert_eq!(pool.in_use(), 1);
        assert!(!pool.contains(NodeId(0)));
        assert!(pool.contains(NodeId(1)));

        // Slot should be reusable
        pool.get_or_allocate(NodeId(2));
        assert_eq!(pool.in_use(), 2);
    }

    #[test]
    fn test_pool_release_batch() {
        let mut pool = ContiguousHistogramPool::new(4, 10);

        for i in 0..4 {
            pool.get_or_allocate(NodeId(i));
        }

        assert_eq!(pool.in_use(), 4);

        pool.release_batch(&[NodeId(0), NodeId(2)]);

        assert_eq!(pool.in_use(), 2);
        assert!(!pool.contains(NodeId(0)));
        assert!(pool.contains(NodeId(1)));
        assert!(!pool.contains(NodeId(2)));
        assert!(pool.contains(NodeId(3)));
    }

    #[test]
    fn test_pool_reset() {
        let mut pool = ContiguousHistogramPool::new(4, 10);

        for i in 0..4 {
            let mut hist = pool.get_or_allocate(NodeId(i));
            hist.add(0, i as f32, 1.0);
        }

        assert_eq!(pool.in_use(), 4);
        assert_eq!(pool.metrics().misses, 4);

        pool.reset();

        assert_eq!(pool.in_use(), 0);
        assert_eq!(pool.available(), 4);
        assert!(!pool.contains(NodeId(0)));

        // Metrics are NOT reset
        assert_eq!(pool.metrics().misses, 4);

        // Can reallocate
        pool.get_or_allocate(NodeId(0));
        assert_eq!(pool.in_use(), 1);
    }

    #[test]
    fn test_pool_metrics() {
        let mut pool = ContiguousHistogramPool::new(2, 10);

        // Miss
        pool.get_or_allocate(NodeId(0));
        assert_eq!(pool.metrics().misses, 1);
        assert_eq!(pool.metrics().hits, 0);

        // Hit
        pool.get(NodeId(0));
        assert_eq!(pool.metrics().hits, 1);

        // Miss
        pool.get_or_allocate(NodeId(1));
        assert_eq!(pool.metrics().misses, 2);

        // Eviction
        pool.get_or_allocate(NodeId(2));
        assert_eq!(pool.metrics().evictions, 1);

        // Peak usage
        assert_eq!(pool.metrics().peak_usage, 2);
    }

    #[test]
    fn test_pool_full_capacity() {
        let mut pool = ContiguousHistogramPool::new(4, 10);

        // Fill exactly to capacity
        for i in 0..4 {
            pool.get_or_allocate(NodeId(i));
        }

        assert_eq!(pool.in_use(), 4);
        assert_eq!(pool.available(), 0);
        assert_eq!(pool.metrics().evictions, 0);

        // One more causes eviction
        pool.get_or_allocate(NodeId(4));
        assert_eq!(pool.in_use(), 4);
        assert_eq!(pool.metrics().evictions, 1);
    }

    #[test]
    fn test_pool_contains() {
        let mut pool = ContiguousHistogramPool::new(4, 10);

        assert!(!pool.contains(NodeId(0)));

        pool.get_or_allocate(NodeId(0));
        assert!(pool.contains(NodeId(0)));

        pool.release(NodeId(0));
        assert!(!pool.contains(NodeId(0)));
    }

    #[test]
    fn test_pool_get_nonexistent() {
        let mut pool = ContiguousHistogramPool::new(4, 10);

        assert!(pool.get(NodeId(0)).is_none());
        assert!(pool.get_mut_no_lru(NodeId(0)).is_none());
    }

    #[test]
    fn test_histogram_slot_operations() {
        let mut pool = ContiguousHistogramPool::new(4, 10);

        let mut hist = pool.get_or_allocate(NodeId(0));
        hist.reset();

        // Add samples
        hist.add(0, 1.0, 0.5);
        hist.add(0, 2.0, 1.0);
        hist.add(1, 3.0, 1.5);

        // Check via immutable view
        let hist = pool.get(NodeId(0)).unwrap();
        assert_eq!(hist.bin_stats(0), (3.0, 1.5, 2));
        assert_eq!(hist.bin_stats(1), (3.0, 1.5, 1));
        assert_eq!(hist.total_grad(), 6.0);
        assert_eq!(hist.total_hess(), 3.0);
        assert_eq!(hist.total_count(), 3);
    }
}
