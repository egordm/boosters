# Development Workflow Guide (Draft)

> **Status**: This is a draft document. Conventions may change as the project evolves.

This document describes conventions and best practices for implementing boosters.

---

## Shell Environment

This project uses **fish shell** as the default terminal. Be mindful of syntax differences:

- Fish does not support heredocs (`<< EOF`)
- Use `set` instead of `export` for environment variables
- String quoting differs from bash/zsh
- Use `and`/`or` instead of `&&`/`||` for command chaining (though `&&`/`||` work in recent fish)

When documenting commands, prefer POSIX-compatible syntax or note fish-specific alternatives.

---

## Core Design Principles

### Zero-Copy

Avoid unnecessary data copies. Design APIs to work with borrowed data where possible:

- Use slices (`&[T]`) instead of owned collections when the callee doesn't need ownership
- Accept generic storage types (`S: AsRef<[T]>`) to support both owned and borrowed data
- Prefer returning views/iterators over allocating new collections
- Use column-major SoA layouts to enable zero-copy column access

### Zero-Cost Abstractions

Abstractions should compile down to the same code as hand-written implementations:

- Use generics and monomorphization over dynamic dispatch where performance matters
- Prefer `impl Trait` over `dyn Trait` in hot paths
- Use `#[inline]` judiciously for small, frequently-called functions
- Design traits to enable compiler optimizations (e.g., known sizes, no allocations)

### Column-Major SoA (Structure of Arrays)

We prefer column-major, Structure-of-Arrays layouts as the default.

**Why:**

- Cache-efficient column iteration for training (histogram building, gradient updates)
- Natural support for multi-output models (each output is a contiguous column)
- SIMD-friendly: contiguous data for vectorization
- Easy parallelization: columns are independent

**When row-major is acceptable:**

- Inference (row-at-a-time prediction)
- Operations that fundamentally require row access (e.g., softmax across outputs)

**Decision process for access patterns:**

1. Can the algorithm work with column-wise access? → Use column-major
2. Does row access overhead matter? → Benchmark to decide
3. Is the row access in a hot loop? → Consider caching or layout conversion

### Generic Reusable Components

When implementing algorithms, identify reusable patterns:

- **Poolers**: Generic LIFO/FIFO pools for reusable buffers (histograms, node queues)
- **Iterators**: Composable iteration patterns over data structures
- **Buffers**: Generic contiguous storage with typed access

Example: A histogram pool for GBTree training should be a generic `Pool<T>` that happens to store histograms, not a histogram-specific implementation.

---

## Project Structure

```text
src/
├── compat/              # Format compatibility (XGBoost, LightGBM)
├── data/                # Data matrix abstractions
├── training/            # Training algorithms
│   ├── gbtree/         # Gradient boosted trees
│   ├── gblinear/       # Gradient boosted linear models
│   ├── objectives/     # Loss functions (gradient computation)
│   └── metrics/        # Evaluation metrics
├── inference/           # Prediction/inference
│   ├── gbtree/
│   └── gblinear/
└── lib.rs
```

---

## RFC Conventions

### Focus on Interfaces

RFCs should prioritize interface design over implementation details:

- **Trait definitions**: Show the public interface in Rust
- **Algorithms**: Describe in language-agnostic pseudocode
- **Design decisions**: Document the "why" behind choices

### Pseudocode Style

Use language-agnostic pseudocode similar to scientific papers:

- `←` for assignment
- `FOR...DO...END FOR` for loops
- `IF...THEN...ELSE...END IF` for conditionals
- `RETURN` for function output
- Mathematical notation where appropriate (∑, ∏, etc.)

Example:

```text
ALGORITHM MyAlgorithm(input)
  FOR each element x IN input DO
    result ← process(x)
  END FOR
  RETURN result
```

### What Goes in RFCs vs Code

| Content | Location |
|---------|----------|
| Trait definitions | RFC |
| Algorithm pseudocode | RFC |
| Design decisions | RFC |
| Implementation details | Code comments |
| Optimizations | Code + benchmarks |
| Edge cases | Code + tests |

---

## Testing Strategy

### Test Organization

```text
src/
├── training/
│   ├── objectives.rs    # Contains #[cfg(test)] mod tests { }
│   └── metrics.rs       # Inline unit tests

tests/
├── integration_*.rs     # End-to-end tests
└── test-cases/          # Reference data
```

### Test Types

| Type | Location | Purpose |
|------|----------|---------|
| Unit tests | Inline `#[cfg(test)] mod tests` | Fast, isolated, test private fns |
| Integration tests | `tests/*.rs` | End-to-end, public API only |
| Reference data | `tests/test-cases/` | Compatibility with XGBoost/LightGBM |

---

## Documentation Strategy

### Where to Put What

| What | Where | When |
|------|-------|------|
| Public API docs | `///` rustdoc | As you implement |
| Internal notes | `//` comments | When non-obvious |
| Module overview | `//!` at top of file | Once module is stable |
| Design rationale | RFC | Before/during implementation |
| Research | `docs/design/research/` | Deep dives |

### Rustdoc Guidelines

- Focus on **what** and **why**, not **how**
- Include examples for public APIs
- Link to RFCs for complex design rationale

---

## Commit Message Convention

Format: `type(scope): description`

Example: `feat(objectives): implement softmax loss gradients`

### Types

- `feat` — new feature
- `fix` — bug fix
- `refactor` — code change that neither fixes nor adds
- `docs` — documentation only
- `test` — adding tests
- `chore` — build, deps, tooling

Include `Refs: RFC-XXXX` in the body when relevant.

---

## Quick Reference: File Locations

| Content | Location |
|---------|----------|
| RFCs | `docs/rfcs/` |
| RFC template | `docs/rfcs/0000-template.md` |
| This guide | `docs/design/CONTRIBUTING.md` |
| Research | `docs/design/research/` |
| Source code | `src/` |
| Integration tests | `tests/` |
| Test data | `tests/test-cases/` |
