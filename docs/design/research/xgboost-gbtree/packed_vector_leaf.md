# Packed Vector Leaf

## ELI13
Some trees output a small list of numbers for each leaf (for example, one number per class). If we have many leaves and the numbers are small, we pack those lists into a single long array so each tree leaf’s vector is stored contiguously. That helps the computer find all numbers quickly when it needs them.

## ELI Grad
Vector (or multi-output) leaf stores a small vector of values for each leaf rather than a single scalar. In runtime layout we flatten the leaf values as a 2D array `(leaf_id, leaf_vector_idx)` into a single contiguous buffer: `leaf_values_flat[leaf_id * vector_len + vector_idx]`. Flattening enables GPU/CPU threads to read contiguous blocks for leaf vectors, improving memory coalescing and cache behavior.

### Example memory layout
For 3 leaves and `vector_len = 4`:
```
leaf_values_flat: [ L0C0, L0C1, L0C2, L0C3, L1C0, L1C1, L1C2, L1C3, L2C0, L2C1, L2C2, L2C3 ]
```
When we get a leaf index `leaf_id` for a row, we gather the contiguous slice `leaf_values_flat[leaf_id*4 : leaf_id*4+4]` and add it to `out[row*n_groups : row*n_groups + 4]`.

### Implementation details
- `leaf_vector_len` equals model `OutputLength()` when `IsVectorLeaf()`.
- For scalar leaves it simplifies to `leaf_vector_len == 1` and the model may store scalar array for memory efficiency.
- When adding leaf values to final predictions, use an efficient vectorized copy/accumulate (SIMD block or GPU orthogonal approach).
- Align the `leaf_values_flat` buffer to 16/32 bytes (depending on SIMD/GPU alignment requirements) for best performance.

## Training vs Inference
- Training: If the model is trained with multi-output vector-leaf trees, the training data structure already contains per-leaf vectors (useful for splits and updates). However, training can use dynamic memory layouts and per-node stats; the packed flat array can be prepared once the tree structure is committed, and it is often not required during updates.
- Inference: Packed vector leaf layout is most beneficial for inference, especially on GPU where contiguous memory reads of the vector slice per leaf are coalesced. Packing can be done at model finalization or at model load time before running predictions.
- Notes: If training uses vector-leaf trees, store both training representation and a packed array for immediate inference performance (conversion overhead occurs once at model save/load time).
