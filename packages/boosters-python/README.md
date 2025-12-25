# Boosters Python

Fast gradient boosting library with native Rust core.

## Installation

```bash
# Development install from workspace root
uv run maturin develop -m packages/boosters-python/Cargo.toml
```

## Quick Start

```python
import boosters as bst
import numpy as np

# Create dataset
X = np.random.rand(100, 10).astype(np.float32)
y = np.random.rand(100).astype(np.float32)

# Model training coming soon...
print(f"boosters version: {bst.__version__}")
```

## Status

This package is under active development. See [RFC-0014](../../docs/rfcs/0014-python-bindings.md) for the API design.

### Progress

- [x] Package structure (Story 1.1)
- [ ] Type stub generation (Story 1.2)
- [ ] Python tooling (Story 1.3)
- [ ] CI pipeline (Story 1.4)
- [ ] Configuration types (Epic 2)
- [ ] Dataset handling (Epic 3)
- [ ] Model training/prediction (Epic 4)
- [ ] scikit-learn integration (Epic 5)

## License

MIT
