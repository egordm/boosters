# RFC-0014: Row Partitioning

- **Status**: Draft
- **Created**: 2024-11-30
- **Updated**: 2024-11-30
- **Depends on**: RFC-0011 (Quantization), RFC-0013 (Split Finding)
- **Scope**: Tracking and updating which rows belong to which tree nodes

## Summary

This RFC defines how rows are assigned to tree nodes and how this assignment is updated
after splits:

1. **Row-to-node mapping**: Tracking which rows belong to which node
2. **Split application**: Partitioning rows based on split decision
3. **Position storage**: Memory-efficient representations

## Motivation

During tree building, we need to know which rows belong to each node:

1. **Histogram building**: Only aggregate gradients for rows in the current node
2. **Split application**: After finding a split, reassign rows to children
3. **Leaf prediction**: Final node assignment determines predictions

The data structure must support:

- Fast iteration over rows in a node (for histogram building)
- Efficient updates after splits (for split application)
- Memory efficiency (millions of rows × thousands of nodes)

## Design

### Overview

```
Row partitioning during tree growth:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Initial: All rows in root (node 0)
┌────────────────────────────────┐
│ rows: [0, 1, 2, 3, 4, 5, 6, 7] │  Node 0 (root)
└────────────────────────────────┘

After split on node 0:
┌──────────────────┐  ┌──────────────────┐
│ rows: [0, 2, 5]  │  │ rows: [1, 3, 4,  │
│                  │  │        6, 7]     │
└──────────────────┘  └──────────────────┘
    Node 1 (left)         Node 2 (right)
```

### Position List Representation

```rust
/// Row partitioning using position lists
/// 
/// Each node has a contiguous slice of row indices.
/// After a split, rows are partitioned within their slice.
pub struct RowPartitioner {
    /// Row indices, grouped by node
    /// positions[node_start[n]..node_start[n+1]] = rows in node n
    positions: Vec<u32>,
    
    /// Start offset for each node's rows
    /// Length: num_nodes + 1
    node_start: Vec<u32>,
    
    /// Number of active nodes
    num_nodes: usize,
}

impl RowPartitioner {
    /// Create partitioner with all rows in root node
    pub fn new(num_rows: u32) -> Self {
        Self {
            positions: (0..num_rows).collect(),
            node_start: vec![0, num_rows],
            num_nodes: 1,
        }
    }
    
    /// Get row indices for a node
    #[inline]
    pub fn node_rows(&self, node: u32) -> &[u32] {
        let start = self.node_start[node as usize] as usize;
        let end = self.node_start[node as usize + 1] as usize;
        &self.positions[start..end]
    }
    
    /// Number of rows in a node
    #[inline]
    pub fn node_size(&self, node: u32) -> u32 {
        self.node_start[node as usize + 1] - self.node_start[node as usize]
    }
    
    /// Apply a split: partition rows into left and right children
    /// Returns (left_node_id, right_node_id)
    pub fn apply_split(
        &mut self,
        node: u32,
        split: &SplitInfo,
        index: &GHistIndexMatrix,
    ) -> (u32, u32) {
        let start = self.node_start[node as usize] as usize;
        let end = self.node_start[node as usize + 1] as usize;
        let rows = &mut self.positions[start..end];
        
        // Partition rows: left rows first, then right rows
        let mid = self.partition_rows(rows, split, index);
        
        // Create two new node entries
        let left_node = self.num_nodes as u32;
        let right_node = left_node + 1;
        
        // Update node_start for new nodes
        self.node_start.push(start as u32 + mid as u32);  // Left end = right start
        self.node_start.push(end as u32);                 // Right end
        
        self.num_nodes += 2;
        
        (left_node, right_node)
    }
    
    /// Partition rows in-place based on split decision
    /// Returns the number of rows going left
    fn partition_rows(
        &self,
        rows: &mut [u32],
        split: &SplitInfo,
        index: &GHistIndexMatrix,
    ) -> usize {
        if split.is_categorical {
            self.partition_categorical(rows, split, index)
        } else {
            self.partition_numerical(rows, split, index)
        }
    }
    
    /// Partition for numerical split
    fn partition_numerical(
        &self,
        rows: &mut [u32],
        split: &SplitInfo,
        index: &GHistIndexMatrix,
    ) -> usize {
        let feature = split.feature;
        let threshold_bin = self.find_threshold_bin(split, index);
        let default_left = split.default_left;
        
        // Dutch national flag partition
        let mut left = 0;
        let mut right = rows.len();
        
        while left < right {
            let row = rows[left];
            let bin = index.get(row, feature);
            
            let goes_left = if bin == 0 {
                // Missing value
                default_left
            } else {
                bin <= threshold_bin
            };
            
            if goes_left {
                left += 1;
            } else {
                right -= 1;
                rows.swap(left, right);
            }
        }
        
        left
    }
    
    /// Partition for categorical split
    fn partition_categorical(
        &self,
        rows: &mut [u32],
        split: &SplitInfo,
        index: &GHistIndexMatrix,
    ) -> usize {
        // Build bitset for O(1) category lookup
        let mut left_cats = [0u64; 4];  // Support up to 256 categories
        for &cat in &split.categories_left {
            let word = (cat / 64) as usize;
            let bit = cat % 64;
            if word < 4 {
                left_cats[word] |= 1u64 << bit;
            }
        }
        
        let feature = split.feature;
        let default_left = split.default_left;
        
        let mut left = 0;
        let mut right = rows.len();
        
        while left < right {
            let row = rows[left];
            let bin = index.get(row, feature);
            
            let goes_left = if bin == 0 {
                default_left
            } else {
                let word = (bin / 64) as usize;
                let bit = bin % 64;
                word < 4 && (left_cats[word] >> bit) & 1 == 1
            };
            
            if goes_left {
                left += 1;
            } else {
                right -= 1;
                rows.swap(left, right);
            }
        }
        
        left
    }
    
    /// Find the bin index corresponding to the threshold
    fn find_threshold_bin(&self, split: &SplitInfo, index: &GHistIndexMatrix) -> u8 {
        let cuts = index.cuts.feature_cuts(split.feature);
        match cuts.binary_search_by(|c| c.partial_cmp(&split.threshold).unwrap()) {
            Ok(idx) => idx as u8,
            Err(idx) => idx.saturating_sub(1) as u8,
        }
    }
}
```

### Alternative: Node Index Array

```rust
/// Row partitioning using node index per row
/// 
/// More memory (4 bytes per row vs ~3 bytes amortized for position list)
/// but enables parallel updates.
pub struct NodeIndexPartitioner {
    /// Node assignment for each row
    /// node_idx[row] = which node this row belongs to
    node_idx: Vec<u32>,
    
    /// Count of rows per node (for histogram pre-allocation)
    node_counts: Vec<u32>,
    
    /// Active leaf nodes (not yet split)
    active_nodes: Vec<u32>,
}

impl NodeIndexPartitioner {
    pub fn new(num_rows: u32) -> Self {
        Self {
            node_idx: vec![0; num_rows as usize],
            node_counts: vec![num_rows],
            active_nodes: vec![0],
        }
    }
    
    /// Apply split: update node assignments in parallel
    pub fn apply_split_parallel(
        &mut self,
        node: u32,
        split: &SplitInfo,
        index: &GHistIndexMatrix,
    ) -> (u32, u32) {
        let left_node = self.node_counts.len() as u32;
        let right_node = left_node + 1;
        
        self.node_counts.push(0);
        self.node_counts.push(0);
        
        let left_count = AtomicU32::new(0);
        let right_count = AtomicU32::new(0);
        
        // Parallel update of node assignments
        self.node_idx.par_iter_mut()
            .enumerate()
            .for_each(|(row, node_ref)| {
                if *node_ref == node {
                    let goes_left = self.evaluate_split(row as u32, split, index);
                    if goes_left {
                        *node_ref = left_node;
                        left_count.fetch_add(1, Ordering::Relaxed);
                    } else {
                        *node_ref = right_node;
                        right_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        
        self.node_counts[left_node as usize] = left_count.load(Ordering::Relaxed);
        self.node_counts[right_node as usize] = right_count.load(Ordering::Relaxed);
        
        (left_node, right_node)
    }
    
    /// Collect rows for a node (for histogram building)
    /// Note: This is O(n) — cache result or use position list instead
    pub fn collect_node_rows(&self, node: u32) -> Vec<u32> {
        self.node_idx.iter()
            .enumerate()
            .filter(|(_, &n)| n == node)
            .map(|(row, _)| row as u32)
            .collect()
    }
    
    fn evaluate_split(&self, row: u32, split: &SplitInfo, index: &GHistIndexMatrix) -> bool {
        let bin = index.get(row, split.feature);
        if bin == 0 {
            split.default_left
        } else if split.is_categorical {
            split.categories_left.contains(&(bin as u32))
        } else {
            // Compare bin to threshold bin
            let threshold_bin = index.cuts.bin_value(split.feature, split.threshold);
            bin <= threshold_bin
        }
    }
}
```

### Hybrid Approach

```rust
/// Hybrid partitioner: position lists + node index for parallelism
pub struct HybridPartitioner {
    /// Primary: position list for fast iteration
    positions: RowPartitioner,
    
    /// Secondary: node index for parallel updates
    node_idx: Vec<u32>,
}

impl HybridPartitioner {
    /// Get rows for histogram building (fast, contiguous)
    pub fn node_rows(&self, node: u32) -> &[u32] {
        self.positions.node_rows(node)
    }
    
    /// Apply split with parallel evaluation, then compact
    pub fn apply_split(
        &mut self,
        node: u32,
        split: &SplitInfo,
        index: &GHistIndexMatrix,
    ) -> (u32, u32) {
        // First: parallel evaluation into node_idx
        let rows = self.positions.node_rows(node);
        let left_node = self.positions.num_nodes as u32;
        let right_node = left_node + 1;
        
        rows.par_iter().for_each(|&row| {
            let goes_left = evaluate_split(row, split, index);
            // Store decision in node_idx
            self.node_idx[row as usize] = if goes_left { left_node } else { right_node };
        });
        
        // Second: serial compaction into position list
        // (Could also be parallel with atomic counters)
        self.positions.apply_split(node, split, index);
        
        (left_node, right_node)
    }
}
```

### Memory Layout

```
Position List layout:
━━━━━━━━━━━━━━━━━━━━

positions: [row_ids for node 0...][row_ids for node 1...][row_ids for node 2...]
            ↑                      ↑                      ↑
node_start: [0,                    n0,                    n0+n1,            n0+n1+n2]

Memory: ~4 bytes per row (for positions)
      + ~4 bytes per node (for node_start)

For 1M rows, 1000 nodes: ~4 MB + 4 KB = ~4 MB


Node Index layout:
━━━━━━━━━━━━━━━━━━

node_idx: [node_id for row 0, node_id for row 1, ...]

Memory: 4 bytes per row

For 1M rows: 4 MB

Trade-off:
- Position list: O(1) iteration, O(n_node) split application
- Node index: O(n_total) iteration (filter), O(n_node) parallel split
```

### Integration with Histogram Building

```rust
/// Efficient histogram building with row partitioning
impl HistogramBuilder {
    /// Build histogram using position list (recommended)
    pub fn build_with_positions(
        &mut self,
        hist: &mut NodeHistogram,
        index: &GHistIndexMatrix,
        grads: &GradientBuffer,
        partitioner: &RowPartitioner,
        node: u32,
    ) {
        let rows = partitioner.node_rows(node);
        self.build(hist, index, grads, rows);
    }
    
    /// Build histogram using node index (slower, use only if needed)
    pub fn build_with_node_index(
        &mut self,
        hist: &mut NodeHistogram,
        index: &GHistIndexMatrix,
        grads: &GradientBuffer,
        partitioner: &NodeIndexPartitioner,
        node: u32,
    ) {
        // Collect rows first (O(n) scan)
        let rows = partitioner.collect_node_rows(node);
        self.build(hist, index, grads, &rows);
    }
}
```

### Sampling Integration

```rust
/// Row partitioner with sampling support
impl RowPartitioner {
    /// Create partitioner with sampled rows only
    pub fn with_sampling(num_rows: u32, sample_mask: &[bool]) -> Self {
        let positions: Vec<u32> = (0..num_rows)
            .filter(|&r| sample_mask[r as usize])
            .collect();
        let num_sampled = positions.len() as u32;
        
        Self {
            positions,
            node_start: vec![0, num_sampled],
            num_nodes: 1,
        }
    }
    
    /// Create with GOSS sampling (keep top gradients + random others)
    pub fn with_goss(
        num_rows: u32,
        grads: &GradientBuffer,
        top_rate: f32,
        other_rate: f32,
    ) -> Self {
        // Sort rows by gradient magnitude
        let mut indexed: Vec<_> = (0..num_rows)
            .map(|r| (r, grads.grad(r).abs()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let n_top = ((num_rows as f32) * top_rate) as usize;
        let n_other = ((num_rows as f32) * other_rate) as usize;
        
        let mut positions: Vec<u32> = indexed[..n_top].iter().map(|(r, _)| *r).collect();
        
        // Randomly sample from remaining
        let mut rng = rand::thread_rng();
        let others: Vec<_> = indexed[n_top..].choose_multiple(&mut rng, n_other)
            .map(|(r, _)| *r)
            .collect();
        positions.extend(others);
        
        let num_sampled = positions.len() as u32;
        
        Self {
            positions,
            node_start: vec![0, num_sampled],
            num_nodes: 1,
        }
    }
}
```

## Design Decisions

### DD-1: Position List vs Node Index

**Context**: Two main approaches for tracking row-to-node mapping.

**Options considered**:

1. **Position list**: Contiguous row IDs per node, like XGBoost
2. **Node index**: Node ID per row, enables parallel updates
3. **Hybrid**: Both, for different operations

**Decision**: Position list as primary, with optional node index for parallel splits.

**Rationale**:

- Histogram building dominates runtime, needs fast row iteration
- Position list gives O(1) access to node's rows (contiguous slice)
- Node index requires O(n) scan to collect rows
- Parallel split application can use node index if beneficial

### DD-2: In-Place Partitioning

**Context**: How to partition rows after a split.

**Options considered**:

1. **In-place swap**: Dutch flag partition within existing slice
2. **Copy out**: Create new vectors for left and right
3. **Stable partition**: Preserve relative order (not needed)

**Decision**: In-place swap partition.

**Rationale**:

- No extra memory allocation
- O(n) swaps, cache-friendly
- Order of rows within a node doesn't matter
- XGBoost uses this approach

### DD-3: Categorical Lookup Structure

**Context**: How to check if a category goes left.

**Options considered**:

1. **Linear search**: O(k) per row
2. **HashSet**: O(1) amortized but allocation
3. **Inline bitset**: O(1) with stack-allocated array

**Decision**: Inline bitset for small cardinality, HashSet for large.

**Rationale**:

- Most categoricals have < 256 categories (fits in 32 bytes as bitset)
- Bitset avoids allocation and has excellent cache behavior
- Fall back to HashSet for rare high-cardinality cases

## Integration

| Component | Integration Point | Notes |
|-----------|-------------------|-------|
| RFC-0011 (Quantization) | `QuantizedMatrix` | Used for split evaluation |
| RFC-0012 (Histograms) | `node_rows()` | Provides row lists for building |
| RFC-0013 (Split Finding) | `SplitInfo` | Applied by partitioner |
| RFC-0015 (Tree Growing) | Coordinates splits | Calls apply_split() |
| RFC-0017 (Sampling) | Sampling integration | GOSS/random sampling support |

### Integration with Existing Code

- **New module**: `src/training/partition.rs` for `RowPartitioner`
- **`src/data/sparse.rs`**: CSCMatrix pattern could inform sparse partitioning if needed
- **Rayon integration**: Use `par_iter_mut` for parallel node assignment updates

## Open Questions

1. **Block partitioning**: **Yes** — process in cache-sized blocks (e.g., 64KB) for better locality on large nodes.

2. **Lazy compaction**: **Yes** — defer position list compaction. Use extra memory if it enables better parallelism or simpler code.

3. **GPU partitioning**: Radix-sort based partitioning for GPU. **Defer** to GPU RFC-0022.

## Future Work

- [ ] Block-based partitioning for large nodes
- [ ] Parallel stable partition
- [ ] GPU-friendly radix partitioning

## References

- [XGBoost row_set.h](https://github.com/dmlc/xgboost/blob/master/src/common/row_set.h)
- [LightGBM data_partition.hpp](https://github.com/microsoft/LightGBM/blob/master/src/treelearner/data_partition.hpp)
- [Feature Overview](../FEATURE_OVERVIEW.md) - Priority and design context

## Changelog

- 2024-11-30: Initial draft
