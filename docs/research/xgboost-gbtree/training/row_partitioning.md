# Row Partitioning

## Overview

After a split is applied, we need to assign each row to either the left or right child.
Row partitioning maintains which rows belong to which node as the tree grows.

## The Data Structure: RowSetCollection

XGBoost uses `RowSetCollection` to track row membership per node:

```cpp
class RowSetCollection {
  std::vector<bst_idx_t> row_indices_;  // All row indices
  std::vector<Elem> elem_;              // Per-node: begin/end pointers
  
  struct Elem {
    bst_idx_t* begin;
    bst_idx_t* end;
    bst_node_t node_id;
  };
};
```

**Key property**: Row indices for each node are contiguous in `row_indices_`.
This enables efficient iteration over rows in a node.

```text
Initial (root node 0):
  row_indices_ = [0, 1, 2, 3, 4, 5, 6, 7]
  elem_[0] = {begin=0, end=8, node_id=0}

After split (node 0 → left=1, right=2):
  row_indices_ = [0, 2, 4, 6,  1, 3, 5, 7]  // reordered
  elem_[1] = {begin=0, end=4, node_id=1}   // even rows go left
  elem_[2] = {begin=4, end=8, node_id=2}   // odd rows go right
```

## Partitioning Algorithm

### Single-Threaded Version

```cpp
void Partition(rows, split_condition, gmat, tree) {
  auto* left_ptr = left_begin;
  auto* right_ptr = right_begin;
  
  for (row_idx : rows) {
    // Get bin index for split feature
    bin_idx = gmat.GetBinIndex(row_idx, split_feature);
    
    // Compare against split threshold
    bool go_left;
    if (bin_idx < 0) {
      // Missing value: use default direction
      go_left = tree.DefaultLeft(node_id);
    } else {
      go_left = (bin_idx <= split_bin);
    }
    
    // Assign row to appropriate child
    if (go_left) {
      *left_ptr++ = row_idx;
    } else {
      *right_ptr++ = row_idx;
    }
  }
}
```

### Parallel Version

XGBoost uses `PartitionBuilder` for parallel partitioning:

```text
┌─────────────────────────────────────────────────────────────────┐
│  1. Split row range into blocks                                 │
│     Block 0: rows [0, 1024)                                     │
│     Block 1: rows [1024, 2048)                                  │
│     ...                                                         │
├─────────────────────────────────────────────────────────────────┤
│  2. Each thread partitions its blocks (thread-local buffers)    │
│     Thread 0: Block 0 → local_left_0, local_right_0             │
│     Thread 1: Block 1 → local_left_1, local_right_1             │
├─────────────────────────────────────────────────────────────────┤
│  3. Calculate global offsets (prefix sum)                       │
│     left_offset[0] = 0                                          │
│     left_offset[1] = local_left_0.size()                        │
│     left_offset[2] = left_offset[1] + local_left_1.size()       │
├─────────────────────────────────────────────────────────────────┤
│  4. Copy thread-local results to final positions                │
│     memcpy(result + left_offset[tid], local_left[tid])          │
└─────────────────────────────────────────────────────────────────┘
```

### PartitionBuilder Implementation

```cpp
template <size_t kBlockSize>
class PartitionBuilder {
  struct BlockInfo {
    size_t n_left;   // Count going left
    size_t n_right;  // Count going right  
  };
  
  std::vector<BlockInfo> blocks_;
  std::vector<size_t> left_right_nodes_sizes_;  // Per-node counts
  std::vector<std::vector<bst_idx_t>> left_data_;   // Thread-local left
  std::vector<std::vector<bst_idx_t>> right_data_;  // Thread-local right
  
  void Partition(node, rows, split_condition, gmat, tree) {
    // 1. Local partitioning (parallel)
    ParallelFor(blocks, [&](block_idx) {
      auto& block = blocks_[block_idx];
      // ... partition block into left/right
    });
    
    // 2. Calculate offsets (sequential)
    CalculateRowOffsets();
    
    // 3. Merge to final array (parallel)
    ParallelFor(blocks, [&](block_idx) {
      MergeToArray(block_idx, result);
    });
  }
};
```

## Split Condition

Finding the split bin requires the histogram cuts:

```cpp
// Find which bin contains the split threshold
int32_t FindSplitCondition(node, tree, gmat) {
  auto fidx = tree.SplitIndex(node);
  auto split_value = tree.SplitCond(node);
  
  auto& cut_ptrs = gmat.cut.Ptrs();
  auto& cut_vals = gmat.cut.Values();
  
  // Binary search for split_value in cuts
  for (bin = cut_ptrs[fidx]; bin < cut_ptrs[fidx + 1]; ++bin) {
    if (split_value == cut_vals[bin]) {
      return bin;
    }
  }
  return -1;  // Not found (shouldn't happen)
}
```

## Bin Comparison Logic

```cpp
// Given a row's bin index and the split bin
bool GoLeft(bin_idx, split_bin, default_left) {
  if (bin_idx < 0) {
    // Missing value
    return default_left;
  }
  // Row goes left if its bin <= split bin
  return bin_idx <= split_bin;
}
```

Note: This assumes the split is "≤ threshold goes left", which is XGBoost's convention.

## Multi-Node Partitioning

When processing multiple nodes at the same depth:

```cpp
void UpdatePosition(gmat, nodes, tree) {
  // Find split conditions for all nodes
  std::vector<int32_t> split_conditions(nodes.size());
  FindSplitConditions(nodes, tree, gmat, &split_conditions);
  
  // Blocked space: (node, row_block) pairs
  BlockedSpace2d space(nodes.size(), 
    [&](node_in_set) { return row_set[nodes[node_in_set].nid].size(); },
    kBlockSize);
  
  // Phase 1: Local partitioning
  ParallelFor2d(space, [&](node_in_set, row_range) {
    auto nid = nodes[node_in_set].nid;
    auto split_cond = split_conditions[node_in_set];
    partition_builder.Partition(node_in_set, row_range, split_cond, ...);
  });
  
  // Phase 2: Calculate offsets
  partition_builder.CalculateRowOffsets();
  
  // Phase 3: Merge results
  ParallelFor2d(space, [&](node_in_set, row_range) {
    partition_builder.MergeToArray(node_in_set, row_range, ...);
  });
  
  // Phase 4: Update row set collection
  AddSplitsToRowSet(nodes, tree);
}
```

## Memory Layout After Partitioning

```text
Before partitioning node 0:
  row_indices: [r0, r1, r2, r3, r4, r5, r6, r7]
               └──────────── node 0 ────────────┘

After partitioning:
  row_indices: [r0, r2, r4, r6, r1, r3, r5, r7]
               └── node 1 ──┘ └── node 2 ──┘

elem_[1] = {begin=0, end=4}   // rows that went left
elem_[2] = {begin=4, end=8}   // rows that went right
```

## Performance Considerations

### Memory Access Pattern

- Reading bin indices: Potentially random access if rows are scattered
- Writing row indices: Sequential within each thread's buffer

### Cache Efficiency

- Block size (typically 2048) chosen to fit in L2 cache
- Larger blocks = fewer synchronization points
- Smaller blocks = better load balancing

### SIMD Potential

- Bin comparison can be vectorized
- Gather operations for loading bin indices
- Compress/store operations for writing results

## Considerations for booste-rs

### What We Need

1. **RowSetCollection**: Track row membership per node
2. **PartitionBuilder**: Parallel partitioning
3. **Split condition lookup**: Map split to bin index

### Potential Simplifications

1. **Start single-threaded**: Get correctness first
2. **Simple array storage**: Don't optimize memory layout initially
3. **Linear split condition search**: Binary search can come later

### Potential Improvements

1. **SIMD partitioning**: Vectorized comparison and scatter
2. **Cache-oblivious blocking**: Auto-tune block sizes
3. **In-place partitioning**: Reduce memory allocation

## Source Code References

| Component | XGBoost Source |
|-----------|----------------|
| CommonRowPartitioner | `src/tree/common_row_partitioner.h` |
| PartitionBuilder | `src/common/partition_builder.h` |
| RowSetCollection | `src/common/row_set.h` |
| UpdatePosition | `src/tree/common_row_partitioner.h` |
