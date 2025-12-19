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
from boosters import GBDTBooster

# Create training data
X = np.random.randn(1000, 10).astype(np.float32)
y = np.random.randn(1000).astype(np.float32)

# Train model (sklearn-style API)
model = GBDTBooster(n_estimators=100, max_depth=6, learning_rate=0.1)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Feature importance
importance = model.feature_importance("gain")

# SHAP values
shap_values = model.shap_values(X[:10])

# With categorical features (indices 0 and 3 are categorical)
model.fit(X, y, categorical_features=[0, 3])
```

## License

MIT OR Apache-2.0
