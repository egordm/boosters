# Design Documentation

Architecture and design decisions for booste-rs.

## Contents

| Document | Description |
|----------|-------------|
| [rfcs/](./rfcs/) | Request for Comments - detailed specifications |
| [research/](./research/) | Research notes on XGBoost/LightGBM internals |
| [definitions_and_principles.md](./definitions_and_principles.md) | Core terminology and design principles |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | Contribution guidelines |

## Architecture Overview

```
booste-rs/
├── src/
│   ├── data/           # Matrix types, binned datasets
│   ├── inference/      # Prediction (GBDT, GBLinear)
│   ├── training/       # Training (GBDT, GBLinear, objectives, metrics)
│   ├── compat/         # XGBoost model loading
│   └── repr/           # Internal representations
├── benches/            # Performance benchmarks
└── tests/              # Integration tests
```

## Key Design Decisions

### 1. Histogram-Based Training
We use the histogram-based approach (like LightGBM) rather than exact greedy (like original XGBoost). This provides O(#bins) split finding vs O(#samples).

### 2. Feature-Parallel Strategy
For histogram building, we parallelize across features rather than rows. This provides better cache utilization for dense data.

### 3. Ordered Gradients
Gradients are pre-gathered into partition order before histogram building, converting random reads to sequential access.

### 4. XGBoost Compatibility
We load XGBoost JSON models directly and produce identical predictions, enabling drop-in replacement for inference workloads.

## RFCs

See [rfcs/README.md](./rfcs/README.md) for the full specification index.
