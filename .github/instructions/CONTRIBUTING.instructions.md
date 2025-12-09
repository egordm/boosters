---
applyTo: '**'
---
# Development Workflow Guide

This document describes conventions and best practices for implementing booste-rs.

---

## API Stability

**This library has no users yet.** Breaking API changes are allowed and encouraged
when they improve the design. Do not hesitate to:

- Rename types, functions, or modules for clarity
- Change function signatures to be more ergonomic
- Remove deprecated code immediately (no deprecation period needed)
- Restructure modules when it makes the codebase cleaner
- Refactor tests to use a different method because a convenience method has been removed to favor clean code.

When the library reaches 1.0 or gains external users, we'll adopt semantic versioning
and be more careful about breaking changes.

---

## Project Organization

### Epics, Stories, and Tasks

We use a lightweight agile structure:

| Level | Description | Location |
|-------|-------------|----------|
| **Epic** | Large feature area (e.g., "GBLinear Support") | `docs/backlog/<epic>.md` |
| **Story** | Deliverable chunk within an epic | Numbered sections in epic file |
| **Task** | Individual work item within a story | Checklist items (1.1, 1.2, etc.) |

### Backlog Structure

```
docs/
├── ROADMAP.md              # High-level priorities & current focus
├── backlog/
│   ├── 01-gbtree-inference.md # Epic: GBTree Inference (complete)
│   ├── 02-gblinear.md         # Epic: GBLinear Support (active)
│   └── 99-future.md           # Backlog of future epics
└── design/
    └── rfcs/               # Design documents
```

**Key principle**: Each epic file uses relative numbering (Story 1, 2, 3...) so
adding/removing epics doesn't require renumbering other files.

---

## Development Loop

For each task in a story:

```
┌─────────────────────────────────────────────────────────────┐
│  1. Read relevant RFC section                                │
│  2. Implement types/functions                                │
│  3. Write unit tests (inline)                                │
│  4. Write rustdoc for public items                           │
│  5. If design changes needed → update RFC, note why          │
│  6. Integration test at story boundary                       │
│  7. Commit with clear message referencing story/task         │
└─────────────────────────────────────────────────────────────┘
```

---

## RFC Updates

RFCs are **living documents** but should remain **stable** after acceptance.

### When to Update RFCs

| Situation | Action |
|-----------|--------|
| Breaking design change | Update RFC, add changelog entry |
| New design decision discovered | Add to DD section with `[DECIDED]` |
| Ambiguity that required a choice | Add clarification |
| Minor implementation detail | Document in code comments instead |
| Performance tweak | Document in code or benchmarks |
| Things that "just work" | No update needed |

### RFC Changelog Format

If you update an accepted RFC, add a changelog at the bottom:

```markdown
## Changelog

- 2024-11-28: DD-8 added — chose X over Y based on implementation experience
- 2024-12-01: Clarified SplitCondition default_left semantics
```

### Decision Flow

```
┌─────────────────────────────────────────────────────┐
│  "The RFC doesn't cover X..."                        │
│                                                      │
│  Is X a fundamental design choice?                   │
│    YES → Update RFC with new DD                      │
│    NO  → Document in code comments                   │
│                                                      │
│  Will X affect other components?                     │
│    YES → Update RFC, check integration table         │
│    NO  → Just implement and document locally         │
└─────────────────────────────────────────────────────┘
```

---

## Documentation Strategy

### Where to Put What

| What | Where | When |
|------|-------|------|
| Public API docs | `///` rustdoc on pub items | As you implement |
| Internal notes | `//` comments | When non-obvious |
| Module overview | `//!` at top of file | Once module is stable |
| Design rationale | RFC or `docs/design/research/` | Before/during implementation |
| Quick thoughts | `docs/design/NOTES.md` | Anytime |

### Rustdoc Guidelines

- Focus on **what** and **why**, not **how** (code shows how)
- Include examples for public APIs
- Link to RFCs for complex design rationale

```rust
/// A single tree in the ensemble.
///
/// Uses Structure-of-Arrays layout for cache-friendly traversal.
/// See RFC-0002 for design rationale.
///
/// # Example
///
/// ```
/// let tree = SoATreeStorage::new(nodes);
/// let leaf = tree.predict_row(&features);
/// ```
pub struct SoATreeStorage<L: LeafValue> { /* ... */ }
```

---

## Testing Strategy

### Test Utilities

The `booste_rs::testing` module provides common assertion helpers:

```rust
use booste_rs::{assert_approx_eq, assert_approx_eq_f64};
use booste_rs::testing::{
    assert_slice_approx_eq, assert_predictions_eq,
    DEFAULT_TOLERANCE, DEFAULT_TOLERANCE_F64,
};

#[test]
fn example_test() {
    // Use macros for simple float comparisons
    assert_approx_eq!(1.0f32, 1.0001f32, 0.001);
    
    // Use functions for slice/prediction comparisons
    assert_slice_approx_eq(&actual, &expected, DEFAULT_TOLERANCE, "context");
}
```

For integration tests, use `tests/test_data.rs` for test case loading helpers.

### Test Organization

```
src/
├── testing.rs          # Test utilities (assertion helpers, tolerances)
├── trees/
│   ├── node.rs         # Contains #[cfg(test)] mod tests { } at bottom
│   ├── storage.rs      # Same — unit tests inline
│   └── mod.rs
│
tests/
├── inference_model.rs      # Forest/tree inference tests
├── inference_xgboost.rs    # XGBoost compatibility tests
├── training_gblinear/      # GBLinear training tests
│   └── *.rs
├── test_data.rs            # Test case loading utilities
└── test-cases/
    └── xgboost/            # Reference models + expected outputs
```

**Why inline unit tests?** This is the Rust idiom. Benefits:

- Tests live next to the code they test
- Can test private functions directly
- Easier to keep in sync when refactoring

The `tests/` folder is only for **integration tests** that use the crate as an external consumer would.

### Test Types

| Type | Location | Purpose |
|------|----------|---------|
| Unit tests | Inline `#[cfg(test)] mod tests` in `src/` | Fast, isolated, can test private fns |
| Integration tests | `tests/*.rs` | End-to-end, public API only |
| Reference data | `tests/test-cases/` | Models + expected outputs from Python |

### Testing Workflow

1. **Unit tests**: Write as you implement each function
2. **Integration tests**: Add at story boundaries
3. **Reference data**: Generate with Python, commit to repo

```rust
// Inline unit test example
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_leaf_accumulates() {
        let mut acc = ScalarLeaf(0.0);
        acc.accumulate(&ScalarLeaf(1.5));
        acc.accumulate(&ScalarLeaf(2.5));
        assert_eq!(acc.0, 4.0);
    }
}
```

---

## Implementation Research

Use `docs/design/research/` for deep dives:

- XGBoost/LightGBM internals exploration
- Performance experiments
- Algorithm alternatives

**Don't** put implementation research in RFCs — those should stay high-level.

Example structure:

```
docs/design/research/
├── xgboost-gbtree/         # GBTree-specific research
├── gblinear/               # GBLinear-specific research
└── lightgbm/               # Future LightGBM research
```

---

## Commit Message Convention

```
<type>(<scope>): <description> [<epic>/<story>.<task>]

<body>

Refs: RFC-XXXX
```

### Types

- `feat` — new feature
- `fix` — bug fix
- `refactor` — code change that neither fixes nor adds
- `docs` — documentation only
- `test` — adding tests
- `chore` — build, deps, tooling

### Examples

```
feat(linear): implement LinearModel struct [gblinear/1.1]

- Box<[f32]> weight storage
- Weight indexing for feature × group
- predict_row() dot product

Refs: RFC-0008
```

```
fix(compat): handle XGBoost default_left edge case [gbtree/1.4]

XGBoost JSON uses 0/1 for default direction, not boolean.
Updated parser to handle numeric values.

Refs: RFC-0007
```

---

## Quick Reference: File Locations

| Content | Location |
|---------|----------|
| High-level roadmap | `docs/ROADMAP.md` |
| Epic details | `docs/backlog/<epic>.md` |
| RFCs (design docs) | `docs/design/rfcs/0XXX-*.md` |
| RFC template | `docs/design/rfcs/TEMPLATE.md` |
| Research & deep dives | `docs/design/research/` |
| Scratch notes | `docs/design/NOTES.md` |
| This guide | `docs/design/CONTRIBUTING.md` |
| Benchmarks | `docs/benchmarks/` |
| Benchmark template | `docs/benchmarks/TEMPLATE.md` |
| Source code | `src/` |
| Test utilities | `src/testing.rs` |
| Integration tests | `tests/` |
| Test data loader | `tests/test_data.rs` |
| Test data | `tests/test-cases/` |
| Python data generation | `tools/data_generation/` |

---

## Summary

1. **Epics organize work** — one file per major feature, relative numbering
2. **RFCs are stable** — update only for significant design changes
3. **Document in code** — rustdoc for public APIs, comments for internals
4. **Test as you go** — don't batch testing at the end
5. **Clear commits** — reference epic/story/task and RFCs
