# Implementation Notes

Scratch space for thoughts, questions, and observations during implementation.

Periodically review and promote important items to:

- RFCs (design decisions)
- `design/analysis/` (research findings)
- Code comments (implementation details)
- GitHub issues (bugs, TODOs)

---

## Open Questions

Things to figure out or decide later.

- [ ] _Example: Should `SoATreeStorage` store node count or compute from array length?_

---

## Ideas & Observations

Thoughts that might be useful later.

### Performance

<!-- Notes about performance observations, potential optimizations -->

### API Design

<!-- Thoughts about API ergonomics, naming, etc. -->

### XGBoost Behavior

<!-- Observations about XGBoost quirks, edge cases, undocumented behavior -->

---

## Implementation Blockers

Things that are blocking progress.

<!-- Use format: What's blocked | Why | Potential solutions -->

---

## Decisions Made (Not in RFC)

Small decisions that don't warrant RFC updates but should be documented.

<!-- Use format: Decision | Rationale | Date -->

| Decision | Rationale | Date |
|----------|-----------|------|
| _Example: Use `Box<[T]>` over `Vec<T>` for immutable arrays_ | _Signals immutability, slightly smaller_ | _2024-11-27_ |

---

## Code Snippets

Useful code patterns or experiments to reference later.

```rust
// Example: fast NaN check
fn has_nan(slice: &[f32]) -> bool {
    slice.iter().any(|x| x.is_nan())
}
```

---

## Links & References

Useful resources discovered during implementation.

- [XGBoost model format docs](https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html)
- [bincode usage examples](https://docs.rs/bincode/latest/bincode/)

---

## Session Log

Brief notes from implementation sessions (optional).

### YYYY-MM-DD

<!-- What you worked on, what you learned, what's next -->
