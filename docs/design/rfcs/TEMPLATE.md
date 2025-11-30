# RFC Template

Use this template when creating a new RFC.

**Before writing an RFC:**
1. Do research first — document findings in `../research/`
2. Discuss the approach informally if needed
3. RFCs are for **design decisions**, not implementation details

---

# RFC-XXXX: [Title]

- **Status**: Draft | Review | Accepted | Active | Implemented | Deprecated
- **Created**: YYYY-MM-DD
- **Updated**: YYYY-MM-DD
- **Depends on**: RFC-NNNN (if any)
- **Scope**: [What aspect of the system this covers]

## Summary

<!-- 2-3 sentences describing what this RFC proposes -->

## Motivation

<!-- Why do we need this? What problem does it solve? -->

## Design

### Overview

<!-- High-level description of the design -->

### Data Structures

<!-- Key types and their relationships -->

```rust
// Example structures
pub struct ExampleType {
    // ...
}
```

### Algorithms

<!-- Key algorithms, with pseudocode if helpful -->

### API

<!-- Public API surface -->

```rust
// Example API
impl ExampleType {
    pub fn new() -> Self { /* ... */ }
    pub fn do_something(&self) -> Result<(), Error> { /* ... */ }
}
```

## Design Decisions

Document key choices made during design. Use this format:

### DD-1: [Decision Title]

**Context**: What situation required a decision?

**Options considered**:
1. Option A — description
2. Option B — description

**Decision**: We chose Option X because...

**Consequences**: What are the implications?

<!-- Add more DD-N sections as needed -->

## Integration

<!-- How does this interact with other components? -->

| Component | Integration Point | Notes |
|-----------|------------------|-------|
| RFC-0001 | ... | ... |

## Open Questions

<!-- Unresolved issues that need discussion -->

1. **Question**: ...
   - Option A
   - Option B

## Future Work

<!-- Out of scope for this RFC but worth noting -->

- [ ] Potential future enhancement 1
- [ ] Potential future enhancement 2

## References

<!-- Links to research, external docs, related code -->

- [Research doc](../research/example.md)
- [XGBoost source](link)
- [Paper](link)

## Changelog

<!-- Add entries when updating an accepted RFC -->

- YYYY-MM-DD: Initial draft
