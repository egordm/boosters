# Development Workflow Guide

This document describes conventions and best practices for implementing booste-rs.

---

## Project Organization

### Epics, Stories, and Tasks

We use a lightweight agile structure:

| Level | Description | Location |
|-------|-------------|----------|
| **Epic** | Large feature area (e.g., "GBLinear Support") | `backlog/<epic>.md` |
| **Story** | Deliverable chunk within an epic | Numbered sections in epic file |
| **Task** | Individual work item within a story | Checklist items (1.1, 1.2, etc.) |

### Backlog Structure

```text
docs/design/
├── ROADMAP.md              # High-level priorities & current focus
├── backlog/
│   ├── gbtree-inference.md # Epic: GBTree Inference (complete)
│   ├── gblinear.md         # Epic: GBLinear Support (active)
│   └── future.md           # Backlog of future epics
└── rfcs/                   # Design documents
```

**Key principle**: Each epic file uses relative numbering (Story 1, 2, 3...) so
adding/removing epics doesn't require renumbering other files.

### When to Create a New Epic

- Starting a significant new feature area
- Moving an item from `future.md` to active development
- Breaking up a large epic that's grown too complex

---

## Development Loop

For each task in a story:

```text
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

```text
┌─────────────────────────────────────────────────────┐
│  "The RFC doesn't cover X..."                       │
│                                                     │
│  Is X a fundamental design choice?                  │
│    YES → Update RFC with new DD                     │
│    NO  → Document in code comments                  │
│                                                     │
│  Will X affect other components?                    │
│    YES → Update RFC, check integration table        │
│    NO  → Just implement and document locally        │
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
| Design rationale | RFC or `design/research/` | Before/during implementation |
| Quick thoughts | `design/NOTES.md` | Anytime |

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

### Test Organization

```text
src/
├── trees/
│   ├── node.rs         # Contains #[cfg(test)] mod tests { } at bottom
│   ├── storage.rs      # Same — unit tests inline
│   └── mod.rs
│
tests/
├── predict_xgboost.rs  # Integration: load model, predict, compare to Python
└── test-cases/
    └── xgboost-models/ # Reference models + expected outputs
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

Use `design/research/` for deep dives:

- XGBoost/LightGBM internals exploration
- Performance experiments
- Algorithm alternatives

**Don't** put implementation research in RFCs — those should stay high-level.

Example structure:

```text
design/research/
├── xgboost-gbtree/         # GBTree-specific research
├── gblinear/               # GBLinear-specific research
└── lightgbm/               # Future LightGBM research
```

---

## Commit Message Convention

```text
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

```text
feat(linear): implement LinearModel struct [gblinear/1.1]

- Box<[f32]> weight storage
- Weight indexing for feature × group
- predict_row() dot product

Refs: RFC-0008
```

```text
fix(compat): handle XGBoost default_left edge case [gbtree/1.4]

XGBoost JSON uses 0/1 for default direction, not boolean.
Updated parser to handle numeric values.

Refs: RFC-0007
```

---

## Quick Reference: File Locations

| Content | Location |
|---------|----------|
| High-level roadmap | `design/ROADMAP.md` |
| Epic details | `design/backlog/<epic>.md` |
| RFCs (design docs) | `design/rfcs/0XXX-*.md` |
| Research & deep dives | `design/research/` |
| Scratch notes | `design/NOTES.md` |
| This guide | `design/CONTRIBUTING.md` |
| Benchmarks | `benchmarks/` |
| Source code | `src/` |
| Integration tests | `tests/` |
| Test data | `tests/test-cases/` |
| Python data generation | `tools/data_generation/` |

---

## Summary

1. **Epics organize work** — one file per major feature, relative numbering
2. **RFCs are stable** — update only for significant design changes
3. **Document in code** — rustdoc for public APIs, comments for internals
4. **Test as you go** — don't batch testing at the end
5. **Clear commits** — reference epic/story/task and RFCs
