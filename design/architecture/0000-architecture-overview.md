# RFC-0000: Architecture Overview

- **Status**: Accepted
- **Created**: 2024-11-24
- **Scope**: High-level system architecture

## Summary

This document provides a birds-eye view of the booste-rs library architecture, establishing the major components, their responsibilities, and how they interact. It serves as a map for navigating the more detailed RFCs.

## Motivation

A clear architectural overview helps:

1. Onboard contributors to the codebase structure
2. Identify component boundaries and interfaces
3. Guide decisions about where new functionality belongs
4. Ensure consistency across the codebase

## Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              booste-rs Library                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                           Public API Layer                               │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │    │
│  │  │   Booster   │  │  DMatrix    │  │   Model     │  │  Predict    │     │    │
│  │  │   (train)   │  │  (data)     │  │  (load/save)│  │  (infer)    │     │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │    │
│  └─────────┼────────────────┼────────────────┼────────────────┼────────────┘    │
│            │                │                │                │                  │
│  ┌─────────┴────────────────┴────────────────┴────────────────┴────────────┐    │
│  │                         Orchestration Layer                              │    │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │    │
│  │  │    Trainer       │  │    Predictor     │  │   Serialization  │       │    │
│  │  │  (GBTree impl)   │  │  (CPU/GPU impl)  │  │   (JSON/Binary)  │       │    │
│  │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘       │    │
│  └───────────┼─────────────────────┼─────────────────────┼─────────────────┘    │
│              │                     │                     │                       │
│  ┌───────────┴─────────────────────┴─────────────────────┴─────────────────┐    │
│  │                          Core Data Structures                            │    │
│  │                                                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │                        Forest Module                             │    │    │
│  │  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │    │    │
│  │  │  │ NodeForest  │───▶│  SoAForest  │───▶│ PackedForest│          │    │    │
│  │  │  │    (AoS)    │    │   (SoA)     │    │    (GPU)    │          │    │    │
│  │  │  └─────────────┘    └─────────────┘    └─────────────┘          │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │                         Tree Module                              │    │    │
│  │  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │    │    │
│  │  │  │  NodeTree   │    │ SoATreeView │    │ ArrayLayout │          │    │    │
│  │  │  │   (nodes)   │    │  (arrays)   │    │  (unrolled) │          │    │    │
│  │  │  └─────────────┘    └─────────────┘    └─────────────┘          │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │                        Data Module                               │    │    │
│  │  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │    │    │
│  │  │  │ DataMatrix  │    │ DenseMatrix │    │SparseMatrix │          │    │    │
│  │  │  │   (trait)   │    │  (f32/u16)  │    │   (sprs?)   │          │    │    │
│  │  │  └─────────────┘    └─────────────┘    └─────────────┘          │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                           Support Modules                                 │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │   │
│  │  │ Visitors │  │  Buffers │  │   SIMD   │  │ Category │  │Objective │   │   │
│  │  │(traverse)│  │ (thread) │  │  (accel) │  │(encoding)│  │  (loss)  │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### Public API Layer

- **Booster**: High-level training API, manages iterations and early stopping
- **DMatrix**: Data abstraction for features, labels, weights
- **Model**: Load/save interface, format conversion
- **Predict**: High-level prediction API with batching

### Orchestration Layer

- **Trainer**: Implements gradient boosting loop, histogram building, tree growing
- **Predictor**: Dispatches to CPU/GPU implementations, manages threading
- **Serialization**: JSON and binary model formats, compatibility with XGBoost

### Core Data Structures

- **Forest Module**: Collection of trees with metadata (see RFC-0001)
- **Tree Module**: Individual tree structures (see RFC-0002)
- **Data Module**: Feature matrices and quantization (future RFC)

### Support Modules

- **Visitors**: Tree traversal abstractions (see RFC-0003)
- **Buffers**: Thread-local buffer pools
- **SIMD**: Portable SIMD operations
- **Category**: Categorical encoding/decoding
- **Objective**: Loss functions and gradients

## Data Flow

### Training Flow

```text
                    ┌─────────────────┐
                    │   Input Data    │
                    │ (features, y)   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    DMatrix      │
                    │  (quantized)    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │   Histogram     │           │    Gradient     │
    │    Builder      │◀──────────│   Calculator    │
    └────────┬────────┘           └────────┬────────┘
             │                             │
             └──────────────┬──────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   Tree Grower   │
                   │  (find splits)  │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   NodeForest    │
                   │  (accumulate)   │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  freeze() →     │
                   │   SoAForest     │
                   └─────────────────┘
```

### Inference Flow

```text
              ┌─────────────────┐
              │  Saved Model    │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   SoAForest     │
              │   (loaded)      │
              └────────┬────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
  ┌───────────┐  ┌───────────┐  ┌───────────┐
  │Single-Row │  │  Block    │  │   GPU     │
  │ Traversal │  │ Traversal │  │  Kernel   │
  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Predictions   │
              └─────────────────┘
```

## Related RFCs

| RFC | Title | Status |
|-----|-------|--------|
| [0001](./0001-forest-data-structures.md) | Forest Data Structures | Accepted |
| [0002](./0002-tree-data-structures.md) | Tree Data Structures | Accepted |
| [0003](./0003-visitor-and-traversal.md) | Visitor and Traversal Patterns | Accepted |
| [0004](./0004-dmatrix.md) | DMatrix and Data Input | Draft |
| 0005 | Threading and Buffer Management | Planned |
| 0006 | Training Pipeline | Planned |
| [0007](./0007-serialization.md) | Serialization | Draft |

## Open Questions

1. **Crate structure**: Should this be a single crate or a workspace with multiple crates (core, training, inference, gpu)?

2. **Feature flags**: How do we handle optional features (GPU, SIMD, serialization formats)?

3. **Compatibility**: What level of XGBoost model compatibility do we target (JSON only? Binary? All versions?)?

## References

- [design/definitions_and_principles.md](../definitions_and_principles.md)
- [design/analysis/design_challenges_and_tradeoffs.md](../analysis/design_challenges_and_tradeoffs.md)
- XGBoost C++ source: `src/gbm/gbtree.cc`, `src/predictor/cpu_predictor.cc`
