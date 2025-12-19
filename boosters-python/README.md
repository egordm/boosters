# Boosters Python

Python bindings for the boosters gradient boosting library.

## Installation

```bash
# Development install
cd boosters-python
maturin develop

# Build wheel
maturin build --release
```

## Usage

```python
import numpy as np
from boosters import GBDTBooster, GBDTParams, Dataset

# Create training data
X = np.random.randn(1000, 10).astype(np.float32)
y = np.random.randn(1000).astype(np.float32)

# Create dataset
dataset = Dataset(X, y)

# Train model
params = GBDTParams(n_estimators=100, max_depth=6, learning_rate=0.1)
model = GBDTBooster.train(params, dataset)

# Make predictions
predictions = model.predict(X)

# Feature importance
importance = model.feature_importance("gain")

# SHAP values
shap_values = model.shap_values(X[:10])
```

## License

MIT OR Apache-2.0
