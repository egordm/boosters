# Tree Growing Strategies

## Overview

XGBoost supports two tree growing strategies that determine the order in which nodes
are expanded:

1. **Depth-wise** (default): Grow all nodes at the same depth before going deeper
2. **Loss-guided** (Best-first): Always expand the node with highest gain

## The Driver

XGBoost uses a `Driver` class that manages the queue of nodes to expand:

```cpp
template <typename ExpandEntryT>
class Driver {
  TrainParam param_;
  bst_node_t num_leaves_ = 1;
  ExpandQueue queue_;  // Priority queue
  
  // Pop nodes to expand next
  std::vector<ExpandEntryT> Pop();
  
  // Check if child can be expanded
  bool IsChildValid(ExpandEntryT const& parent);
};
```

## Depth-Wise Growth

**Strategy**: Process all nodes at depth d before any node at depth d+1.

```text
          0            depth 0: process [0]
         / \
        1   2          depth 1: process [1, 2]
       / \ / \
      3  4 5  6        depth 2: process [3, 4, 5, 6]
```

### Implementation

```cpp
// Comparison function: smaller node_id = smaller depth = higher priority
bool DepthWise(const ExpandEntry& lhs, const ExpandEntry& rhs) {
  return lhs.GetNodeId() > rhs.GetNodeId();  // Favor small node_id
}

std::vector<ExpandEntry> Pop() {
  if (param_.grow_policy == kDepthWise) {
    std::vector<ExpandEntry> result;
    ExpandEntry e = queue_.top();
    int level = e.depth;
    
    // Pop all nodes at the same depth
    while (e.depth == level && !queue_.empty() && result.size() < max_batch) {
      queue_.pop();
      if (e.IsValid(param_, num_leaves_)) {
        num_leaves_++;
        result.push_back(e);
      }
      if (!queue_.empty()) {
        e = queue_.top();
      }
    }
    return result;
  }
  // ...
}
```

### Advantages

- **Parallelism**: All nodes at same depth can be processed together
- **Memory predictability**: Histogram memory is bounded by 2^depth nodes
- **Better for shallow trees**: When max_depth is the limiting factor

### Disadvantages

- **May waste work**: Expands nodes even if gain is low
- **Not optimal**: Doesn't prioritize promising regions

## Loss-Guided Growth (Best-First)

**Strategy**: Always expand the node with highest gain.

```text
          0  (gain=10)     Step 1: expand 0
         / \
        1   2              Step 2: expand 1 (gain=8) or 2 (gain=5)?
       (8) (5)                     → expand 1
       / \
      3   4                Step 3: expand 3 (gain=7), 4 (gain=2), or 2 (gain=5)?
     (7) (2)                       → expand 3
```

### Implementation

```cpp
// Comparison function: higher gain = higher priority
bool LossGuide(const ExpandEntry& lhs, const ExpandEntry& rhs) {
  if (lhs.GetLossChange() == rhs.GetLossChange()) {
    return lhs.GetNodeId() > rhs.GetNodeId();  // Tiebreak by node_id
  }
  return lhs.GetLossChange() < rhs.GetLossChange();  // Favor high gain
}

std::vector<ExpandEntry> Pop() {
  if (param_.grow_policy == kLossGuide) {
    ExpandEntry e = queue_.top();
    queue_.pop();
    
    if (e.IsValid(param_, num_leaves_)) {
      num_leaves_++;
      return {e};  // Return single node
    }
    return {};
  }
  // ...
}
```

### Advantages

- **Optimal for fixed leaf count**: Gets best reduction with max_leaves limit
- **Early stopping potential**: Can stop when gains become small
- **Better for deep trees**: Focuses on promising branches

### Disadvantages

- **Less parallelism**: Typically processes one node at a time
- **Memory growth**: May need to keep more histograms cached
- **Requires max_leaves**: Doesn't work well with just max_depth

## Validity Checks

A node can only be expanded if it passes validity checks:

```cpp
bool IsValid(TrainParam const& param, bst_node_t num_leaves) {
  // Check depth limit
  if (param.max_depth > 0 && depth >= param.max_depth) {
    return false;
  }
  
  // Check leaf count limit
  if (param.max_leaves > 0 && num_leaves >= param.max_leaves) {
    return false;
  }
  
  // Check if split gain is positive
  if (split.loss_chg <= kRtEps) {
    return false;
  }
  
  return true;
}
```

## Expand Entry

Information tracked for each candidate node:

```cpp
struct CPUExpandEntry {
  bst_node_t nid;       // Node index
  int depth;            // Depth in tree
  SplitEntry split;     // Best split found for this node
  
  bool IsValid(TrainParam const& param, bst_node_t num_leaves) const;
  bst_node_t GetNodeId() const { return nid; }
  float GetLossChange() const { return split.loss_chg; }
};
```

## Training Loop

The main training loop using the driver:

```cpp
void UpdateTree(updater, p_fmat, gpair, p_tree) {
  Driver<ExpandEntry> driver{param};
  
  // Initialize root
  driver.Push(updater->InitRoot(p_fmat, gpair, p_tree));
  auto expand_set = driver.Pop();
  
  while (!expand_set.empty()) {
    // Apply splits and update row positions
    std::vector<ExpandEntry> valid_candidates;
    for (auto const& candidate : expand_set) {
      updater->ApplyTreeSplit(candidate, p_tree);
      
      if (driver.IsChildValid(candidate)) {
        valid_candidates.push_back(candidate);
      }
    }
    
    updater->UpdatePosition(p_fmat, p_tree, expand_set);
    
    // Build histograms and evaluate splits for children
    std::vector<ExpandEntry> best_splits;
    if (!valid_candidates.empty()) {
      updater->BuildHistogram(p_fmat, p_tree, valid_candidates, gpair);
      
      for (auto const& candidate : valid_candidates) {
        auto left_child = tree.LeftChild(candidate.nid);
        auto right_child = tree.RightChild(candidate.nid);
        
        best_splits.push_back({left_child, depth+1});
        best_splits.push_back({right_child, depth+1});
      }
      
      updater->EvaluateSplits(p_fmat, p_tree, &best_splits);
    }
    
    // Queue valid children for next iteration
    driver.Push(best_splits);
    expand_set = driver.Pop();
  }
}
```

## Batched Node Processing

For depth-wise growth, XGBoost limits the batch size:

```cpp
size_t max_node_batch_size = 256;  // Process at most 256 nodes together
```

This balances:

- Parallelism (more nodes = more parallel work)
- Memory (more nodes = more histograms needed)
- Synchronization (more nodes = more merging overhead)

## Comparison

| Aspect | Depth-wise | Loss-guided |
|--------|------------|-------------|
| Parallelism | High (batch of nodes) | Low (one node) |
| Memory | Bounded by 2^depth | Can grow unbounded |
| Tree shape | Balanced | Can be imbalanced |
| Best for | max_depth limits | max_leaves limits |
| Default in | XGBoost | LightGBM |

## Considerations for booste-rs

### What We Need

1. **Driver abstraction**: Queue nodes with priority
2. **Both strategies**: Support depth-wise and loss-guided
3. **Validity checks**: max_depth, max_leaves, min_gain

### Potential Simplifications

1. **Start with depth-wise**: Simpler, more parallelism
2. **Fixed batch size**: Don't tune dynamically
3. **Simple priority queue**: std::BinaryHeap in Rust

### Potential Improvements

1. **Adaptive batching**: Adjust based on tree shape
2. **Work stealing**: Balance load across threads
3. **Speculative execution**: Start next level before current finishes

## Source Code References

| Component | XGBoost Source |
|-----------|----------------|
| Driver | `src/tree/driver.h` |
| ExpandEntry | `src/tree/hist/expand_entry.h` |
| UpdateTree | `src/tree/updater_quantile_hist.cc` |
| TrainParam | `src/tree/param.h` |
