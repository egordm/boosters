# Array Tree Layout (Structure of Arrays)

## ELI13
Think of a tree as many little pieces (nodes). Normally, you store each node as a big object with many fields (split feature id, threshold, left child, right child, etc.). In Array Tree Layout (SoA), we store each field as a separate array so the computer reads only what it needs. That makes it faster especially when many CPUs or GPUs read the data at the same time.

## ELI Grad
Array Tree Layout (Structure-of-Arrays) arranges node attributes into separate contiguous arrays: `split_index[]`, `split_thresholds[]`, `left_child[]`, `right_child[]`, `leaf_values[]`. The key advantage is spatial locality for a given attribute across nodes and rows: when the prediction kernel iterates over many trees and rows, it can load contiguous memory and benefit from vectorization or coalesced memory accesses on GPU.

### Why SoA is good on GPU
GPUs are optimized for high-throughput streaming memory accesses where multiple threads read contiguous memory addresses. With SoA, if every thread needs to read `split_index` for many nodes, those reads come from a contiguous slice, enabling coalesced memory reads and reducing memory transactions. Conversely, AoS (array-of-structures) interleaves attributes and causes uncoalesced reads.

### Example memory layout
For a tree with 6 nodes, scalar leaf values:

- SoA (preferred):
```
split_index   = [2, 1, 0, 2, 1, 0]  // u32 per node
split_thresh  = [0.5, 1.2, 0.3, ...] // f32 per node
left_child    = [1, 3, -1, -1,  5, -1]
right_child   = [2, -1, -1, -1, -1, -1]
leaf_values   = [0.1, 0.2, 0.0, -0.3, 0.4, 0.5]
```

- AoS (less efficient):
```
Node[0] = { split_index=2, thresh=0.5, left=1, right=2, leaf=0.1 }
Node[1] = { split_index=1, thresh=1.2, left=3, right=-1, leaf=0.2 }
```

### Traversal pattern & SoA advantage
If multiple threads read `split_index` and `split_thresh` for nodes visited by their rows, SoA lets each GPU warp/tile read `split_index` in one contiguous burst; AoS would force reading interleaved structs causing cache waste and misaligned loads.

### Data structure sketch
- `NodeSoA { split_index: Vec<u32>, split_threshold: Vec<f32>, left: Vec<i32>, right: Vec<i32>, default_left: BitVec }`
- `TreeSoA { nodes: NodeSoA, leaf_values: Vec<f32> }` or flattened to single `Vec<f32>` if vector leaf.

## Training vs Inference
- Training: During training, the model often needs to mutate trees, add nodes, and maintain per-node stats; an AoS-like structure (per-node structs) or a compact dynamic representation can be easier for updaters and histogram operations. However, some training phases (prediction caches, minibatch updates) can benefit from SoA (e.g., faster predictions used for gradient/hessian calculation).
- Inference: SoA is particularly beneficial for inference—both CPU and GPU—as nodes are immutable and can be compacted into read-only arrays allowing SIMD/coalesced loads. Typically, convert tree representation to SoA post-training (or at load time) for best inference performance.
- Notes: Choose the training data structure to simplify mutability and performance, and provide an efficient conversion to SoA at the end of training or at model load time for inference.
