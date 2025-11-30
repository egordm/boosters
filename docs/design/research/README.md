# Research

Deep dives into algorithms, data structures, and external systems.

Research documents are exploratory — understanding how things work before designing
our implementation. They inform RFCs but are not prescriptive.

## Contents

### By Topic

| Folder | Description |
|--------|-------------|
| [gblinear/](gblinear/) | XGBoost's linear booster (elastic net) |
| [xgboost-gbtree/](xgboost-gbtree/) | XGBoost's tree booster and optimizations |

### General

- [data_structures_hierarchy.md](data_structures_hierarchy.md) — Overview of data structure choices
- [design_challenges_and_tradeoffs.md](design_challenges_and_tradeoffs.md) — Key design tensions

## Research vs RFCs

| Research | RFCs |
|----------|------|
| "How does X work?" | "How will we build X?" |
| Exploratory | Prescriptive |
| Can be incomplete | Should be complete |
| External focus | Internal focus |
| Informs decisions | Documents decisions |

## Adding Research

1. Create a folder for the topic if it doesn't exist
2. Add a `README.md` with an overview and table of contents
3. Split into logical files (concepts, algorithms, parameters, etc.)
4. Use ELI5/ELI13/ELI-Grad sections to explain complex topics at different levels
