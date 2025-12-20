# Backlog 10: Stakeholder Feedback Items

**RFC**: Various  
**Priority**: Medium  
**Status**: Backlog

---

## Overview

Follow-up items from stakeholder feedback received during Backlog 09 completion.

---

## Story 10.1: Performance Comparison Benchmarks

Run comparison benchmarks against XGBoost and LightGBM to confirm performance advantage.

**Context**: Previous benchmarks showed booste-rs dominating on training and prediction speed.
Need to re-verify this still holds after the API refactor.

**Note**: Can't trust criterion "regressed" markers since baseline may have been with regression.

**Tasks**:

- [ ] 10.1.1: Run `compare_training` benchmark
- [ ] 10.1.2: Run `compare_prediction` benchmark
- [ ] 10.1.3: Generate benchmark report in docs/benchmarks/
- [ ] 10.1.4: Compare against previous reports

**Definition of Done**:

- Benchmark report showing comparison with XGBoost/LightGBM
- Performance advantage confirmed or regressions identified

---

## Story 10.2: Serialization Format Redesign

Complete removal and redesign of the serialization format.

**Context**: Current format has issues:
- Public API isn't complete, so format is inherently flawed
- Held back making model config non-optional
- Stakeholder wants to write new RFC for redesigned format

**Decision**: Wait for new RFC before implementing.

**Tasks**:

- [ ] 10.2.1: Delete current io/native.rs and related code
- [ ] 10.2.2: Remove storage feature gates
- [ ] 10.2.3: Simplify model structs (config as Option is fine for now)
- [ ] 10.2.4: Wait for new serialization RFC

**Definition of Done**:

- Old serialization code removed
- New RFC written and accepted
- New format implemented

---

## Story 10.3: RFC-10 API Naming

RFC-10 mentions `write_into`/`read_from` instead of `save`/`load`.

**Context**: Current API uses `save(path)`/`load(path)` instead of the
RFC-specified `write_into`/`read_from` pattern.

**Decision**: Review RFC-10 and align naming if appropriate.

**Tasks**:

- [ ] 10.3.1: Review RFC-10 for exact API specification
- [ ] 10.3.2: Decide if renaming is appropriate
- [ ] 10.3.3: If yes, update API names

**Definition of Done**:

- API names consistent with RFC or deviation documented

---

> Note: Story 10.2 and 10.3 may be superseded by the serialization redesign RFC.
