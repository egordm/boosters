---
applyTo: '**'
---
# Development Workflow Guide

This document describes conventions and best practices for implementing booste-rs.

---

## Development Loop

For each milestone in the roadmap:

```
┌─────────────────────────────────────────────────────────────┐
│  1. Read relevant RFC section                                │
│  2. Implement types/functions                                │
│  3. Write unit tests (inline)                                │
│  4. Write rustdoc for public items                           │
│  5. If design changes needed → update RFC, note why          │
│  6. Integration test at milestone boundary                   │
│  7. Commit with clear message referencing milestone          │
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
| Design rationale | RFC or `design/analysis/` | Before/during implementation |
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

```
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
2. **Integration tests**: Add at milestone boundaries
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

Use `design/analysis/` for deep dives:

- XGBoost internals exploration
- Performance experiments
- Algorithm alternatives

**Don't** put implementation research in RFCs — those should stay high-level.

Example structure:

```
design/analysis/
├── xgboost_cpp_inference.md    # How XGBoost C++ does prediction
├── simd_traversal_experiment.md # Performance experiment notes
└── quantized_data_structures/   # Deep dive on quantization
```

---

## Commit Message Convention

```
<type>(<scope>): <description> [M<milestone>]

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
feat(trees): implement SoATreeStorage [M1.2]

- Flat array layout for cache-friendly traversal
- predict_row() traverses to leaf
- Tests for binary tree structure

Refs: RFC-0002
```

```
fix(compat): handle XGBoost default_left edge case [M1.4]

XGBoost JSON uses 0/1 for default direction, not boolean.
Updated parser to handle numeric values.

Refs: RFC-0007
```

```
docs(trees): add module-level rustdoc [M1.2]

Refs: RFC-0002
```

---

## Quick Reference: File Locations

| Content | Location |
|---------|----------|
| Architecture decisions | `design/architecture/0XXX-*.md` |
| Implementation roadmap | `design/ROADMAP.md` |
| Research & deep dives | `design/analysis/` |
| Scratch notes | `design/NOTES.md` |
| This guide | `design/CONTRIBUTING.md` |
| Source code | `src/` |
| Integration tests | `tests/` |
| Test data | `tests/test-cases/` |
| Python data generation | `tools/data_generation/` |

---

## Summary

1. **RFCs are stable** — update only for significant design changes
2. **Document in code** — rustdoc for public APIs, comments for internals
3. **Test as you go** — don't batch testing at the end
4. **Use NOTES.md** — for quick thoughts, promote important bits later
5. **Clear commits** — reference milestones and RFCs
