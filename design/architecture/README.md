# Architecture Design Documents

This folder contains RFC-style design documents for the xgboost-rs library architecture.

## Document Format

Each RFC follows this structure:

- **Status**: See status definitions below
- **Created**: Date
- **Depends on**: Related RFCs
- **Scope**: What aspect of the system this covers

## Status Definitions

| Status | Meaning |
|--------|---------|
| **Draft** | Initial proposal, under development or not yet reviewed |
| **Review** | Ready for review, awaiting feedback |
| **Accepted** | Approved design, not yet implemented |
| **Active** | Accepted and currently being implemented |
| **Implemented** | Fully implemented in code |
| **Deprecated** | No longer applicable, superseded by another RFC |

## RFC Index

| RFC | Title | Status | Summary |
|-----|-------|--------|---------|
| [0000](./0000-architecture-overview.md) | Architecture Overview | Accepted | High-level system architecture and component map |
| [0001](./0001-forest-data-structures.md) | Forest Data Structures | Accepted | `NodeForest`, `SoAForest`, container types |
| [0002](./0002-tree-data-structures.md) | Tree Data Structures | Accepted | Node layout, `NodeTree`, `SoATreeView`, `ArrayTreeLayout` |
| [0003](./0003-visitor-and-traversal.md) | Visitor and Traversal | Accepted | `Visitor` trait, `Predictor`, block traversal |
| [0004](./0004-dmatrix.md) | DMatrix and Data Input | Draft | `DataMatrix` trait, Arrow integration, quantization |
| 0005 | Threading & Buffers | Planned | Thread pools, buffer management |
| 0006 | Training Pipeline | Planned | Histogram building, tree growing |
| [0007](./0007-serialization.md) | Serialization | Draft | Model loading, XGBoost/LightGBM formats, conversion |
| 0008 | SIMD Acceleration | Planned | `std::simd` integration |
| 0009 | GPU Backend | Planned | `wgpu`/CUDA support |

## Dependency Graph

```text
                    ┌─────────────────┐
                    │  0000: Overview │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ 0001: Forest    │ │ 0004: DMatrix   │ │ 0006: Training  │
│                 │ │                 │ │                 │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         ▼                   │                   │
┌─────────────────┐          │                   │
│ 0002: Tree      │◀─ ─ ─ ─ ─┤                   │
│ (related: 0001) │          │                   │
└────────┬────────┘          │                   │
         │                   │                   │
         └─────────┬─────────┘                   │
                   ▼                             │
         ┌─────────────────┐                     │
         │ 0003: Visitor   │◀────────────────────┘
         │ (0001+0002+0004)│
         └────────┬────────┘
                  │
             ┌────┴────┐
             │         │
             ▼         ▼
         ┌───────┐ ┌───────┐
         │ 0005  │ │ 0008  │
         │Buffer │ │ SIMD  │
         └───────┘ └───────┘
```

## How to Use These Documents

1. **Start with RFC-0000** for the big picture
2. **Read dependent RFCs first** (follow the dependency graph)
3. **Each RFC is self-contained** but may reference others
4. **Open Questions** sections identify unresolved design decisions

## Contributing

When updating these documents:

1. Update the status if the RFC moves to a new phase
2. Add superseding RFC number if this is deprecated
3. Update the index table above
4. Keep diagrams text-based (ASCII/Mermaid) for version control

## Key Design Principles

From [definitions_and_principles.md](../definitions_and_principles.md):

1. **Zero-Cost Abstractions**: Rust generics compile to specialized code
2. **Semantic Naming**: Types reflect layout (`SoA`, `Packed`) not lifecycle
3. **Rust-First Safety**: Ownership and lifetimes over runtime checks
4. **Separation of Concerns**: Training vs inference, explicit conversion
5. **Zero-Copy Traversal**: Borrowed views, no structural copies
6. **Explicit Containers**: `Vec`, `Box`, `Arc` are part of the type
