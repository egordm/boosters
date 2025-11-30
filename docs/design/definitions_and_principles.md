# Definitions & Design Principles

This document outlines the terminology, design philosophy, and naming conventions for the project. It emphasizes semantic clarity, Rust idioms, and a clear separation between training and inference concerns.

## 1. Common Terms

- **Feature / Column**: An input variable.
- **Quantized Feature**: A feature value mapped to a discrete bin index (integer).
- **Node**: A decision point in a tree (internal split or terminal leaf).
- **Leaf**: A terminal node containing prediction value(s). Can be scalar (one value) or vector (multi-output).
- **Tree**: A single predictive structure composed of nodes.
- **Forest**: A collection of trees (the model).
- **Group / Output Group**: Target index for multi-class/multi-target models. Trees may be assigned to specific groups.
- **SoA (Structure-of-Arrays)**: Memory layout where node attributes (e.g., split indices, thresholds) are stored in separate contiguous arrays rather than interleaved structs.
- **Packed Layout**: A memory layout optimized for a specific backend (CPU SIMD or GPU), often aligned and contiguous.
- **Visitor**: A traversal abstraction that operates on a view of the forest and performs actions (like accumulation) on leaf hits.
- **Accumulation**: The process of summing leaf values from multiple trees into an output buffer. In gradient boosting, predictions are additive: `output[group] = base_score + Σ leaf_values[tree]`. Accumulation happens per-group for multi-class, and can be vectorized (SIMD) when processing multiple rows or trees. The `accumulate()` method on `LeafValue` adds a single leaf's contribution to the running total.

## 2. Design Principles

1. **Zero-Cost Abstractions**: Leverage Rust's compile-time features (generics, traits, monomorphization) to build high-level abstractions that compile down to optimized machine code. The goal is to match or exceed the performance of the C++ XGBoost implementation while maintaining safety and ergonomics.
2. **Semantic Naming**: Names should reflect memory layout and semantics (e.g., `SoA`, `Grouped`, `Packed`) rather than just lifecycle (e.g., `Runtime`).
3. **Rust-First Safety**: Rely on ownership, lifetimes, and type systems to enforce invariants at compile time. Avoid runtime checks where possible.
4. **Separation of Concerns**:
    - **Training**: Mutable, dynamic, rich metadata (histograms, stats).
    - **Inference**: Immutable, compact, optimized for traversal (SoA, aligned).
    - **Conversion**: Explicit steps to transform training structures into inference layouts.
5. **Zero-Copy Traversal**: Visitors should operate on borrowed views (`&Forest`) without copying structural data.
6. **Explicit Containers**: The container type (`Vec`, `Box`, `Arc`) should be part of the type definition or alias to communicate ownership and mutability clearly.

## 3. Naming & Structuring Conventions

### Semantic Types

Use names that describe *what* the structure is and *how* it is laid out.

- **`NodeForest`**: Mutable, AoS/node-based representation.
- **`SoAForest`**: Inference-oriented, Structure-of-Arrays layout.
- **`SoAGroupedForest`**: SoA layout with explicit group mapping metadata.
- **`LeafVecForest`**: SoA layout where leaves store vectors.
- **`PackedForest`**: Backend-specific (CPU/GPU) packed representation.

**Naming Rationale**: We use layout-descriptive names (Node, SoA, Packed) rather than use-case names (Train, Infer). This avoids semantic lock-in: a "NodeForest" could be used for training, model inspection, or serialization. The name describes the *structure*, not the *purpose*.

### Container Choices & Explicit Typing

Encode the container type in the name or generic parameters to clarify ownership.

- **`Vec<T>`**: Dynamic, resizable (Training).
- **`Box<[T]>`**: Fixed-size, owned, immutable (Inference/Finalized).
- **`Arc<[T]>`**: Shared, thread-safe, immutable (Inference/Serving).
- **`DeviceBuf<T>`**: Backend-specific buffer (GPU).

**Examples:**

- `SoAForestVec` (owned, resizable)
- `SoAForestArc` (shared, immutable)
- `PackedGpuForest` (device memory)

### Traversal & Visitors

- **`ForestView`**: A borrowed, read-only view into a forest.

- **`Visitor`**: A trait for traversal logic (e.g., `visit_leaf`).
- **`BlockVisitor`**: A specialization for processing blocks of rows.

## 4. Architecture: Visitors & Orchestration

- **Orchestration**: The high-level predictor manages data blocks and thread-local buffers. It calls the visitor.
- **Visitor Pattern**: The visitor encapsulates *behavior* (accumulation, transformation). It borrows the `ForestView` and does not own data.
- **Pluggability**: Different storage backends (CPU SoA, GPU Packed) can be traversed by compatible visitors, or visitors can be specialized for layouts.

## 5. Idioms

- Use **Enums** or **Generics** to encode invariants (e.g., `LeafKind::Scalar` vs `LeafKind::Vector`) to reduce runtime branching.

- Favor **Const Generics** for small fixed sizes in hot loops.
- Keep **API Boundaries** explicit: conversion from `NodeForest` to `SoAForest` is a distinct operation.
