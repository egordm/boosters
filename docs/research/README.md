# Research

Deep dives into algorithms, data structures, and techniques used in gradient boosting.

Research documents are **educational** — explaining how gradient boosting algorithms work,
their optimizations, and trade-offs. They inform implementation decisions but are not
prescriptive to our library's specific design.

---

## Contents

### Foundations

| Document | Description |
|----------|-------------|
| [gradient-boosting.md](gradient-boosting.md) | The gradient boosting **method** (theory) |

### Algorithms

| Folder | Description |
|--------|-------------|
| [gbdt/](gbdt/) | Gradient Boosted Decision Trees (training, inference, data structures) |
| [gblinear/](gblinear/) | Linear gradient boosting with elastic net (coordinate descent) |

### Cross-Cutting Topics

| Document | Description |
|----------|-------------|
| [categorical-features.md](categorical-features.md) | Native categorical feature handling |

### Reference

| Document | Description |
|----------|-------------|
| [library-comparison.md](library-comparison.md) | XGBoost vs LightGBM feature comparison |
| [implementation-notes.md](implementation-notes.md) | Key lessons and design decisions |

---

## Document Structure

### Algorithm-Focused Organization

Documentation is organized by **algorithm** (GBDT, GBLinear), not by library. Each
algorithm folder contains:

- `training/` — How models are trained
- `inference/` — How predictions are made  
- `data-structures/` — Key representations (GBDT only)

This reflects that training and inference have different concerns and optimizations.

---

## Quick Navigation

### GBDT Training Pipeline

1. **Quantization**: [gbdt/training/quantization.md](gbdt/training/quantization.md)
   — Convert continuous features to discrete bins

2. **Histogram Building**: [gbdt/training/histogram-training.md](gbdt/training/histogram-training.md)
   — Aggregate gradients per bin, subtraction trick

3. **Split Finding**: [gbdt/training/split-finding.md](gbdt/training/split-finding.md)
   — Evaluate gain for candidate splits

4. **Tree Growth**: [gbdt/training/tree-growth-strategies.md](gbdt/training/tree-growth-strategies.md)
   — Depth-wise vs leaf-wise growth

5. **Sampling**: [gbdt/training/sampling-strategies.md](gbdt/training/sampling-strategies.md)
   — Random subsampling and GOSS

### GBDT Inference

- [gbdt/inference/batch-traversal.md](gbdt/inference/batch-traversal.md) — Batch prediction
- [gbdt/inference/multi-output.md](gbdt/inference/multi-output.md) — Multi-class/multi-target

### GBLinear

- [gblinear/training/](gblinear/training/) — Coordinate descent optimization
- [gblinear/inference/](gblinear/inference/) — Linear prediction

### Special Features

- **Categorical Features**: [categorical-features.md](categorical-features.md)

---

## Research vs RFCs

| Research | RFCs |
|----------|------|
| "How does gradient boosting work?" | "How will we build it?" |
| Algorithm documentation | Design decisions |
| External focus (XGBoost, LightGBM) | Internal focus (booste-rs) |
| Educational | Prescriptive |
| Can cite academic papers | Should be self-contained |

---

## Primary Sources

These documents synthesize information from:

- **XGBoost**: [github.com/dmlc/xgboost](https://github.com/dmlc/xgboost)
  - Primary reference for histogram-based training
  - JSON model format compatibility
  
- **LightGBM**: [github.com/microsoft/LightGBM](https://github.com/microsoft/LightGBM)
  - Leaf-wise growth strategy
  - GOSS sampling
  - Native categorical handling

- **Academic Papers**:
  - Chen & Guestrin (2016): XGBoost: A Scalable Tree Boosting System
  - Ke et al. (2017): LightGBM: A Highly Efficient Gradient Boosting Decision Tree

---

## Adding Research

When adding new research documents:

1. Create in the appropriate folder (or create a new folder for major topics)
2. Use the ELI5/ELI-Grad format where appropriate
3. Reference both XGBoost and LightGBM approaches where applicable
4. Focus on algorithms and concepts, not library-specific implementation details
5. Link from this README's table of contents
