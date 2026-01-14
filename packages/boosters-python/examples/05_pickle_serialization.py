"""Example: Pickle serialization with boosters.

Shows how to save and load models using Python's pickle module.
"""

import pickle

import numpy as np

from boosters import Dataset, GBDTConfig, GBDTModel, Metric, Objective

# Create synthetic data
rng = np.random.default_rng(42)
X = rng.standard_normal((1000, 10), dtype=np.float32)
y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(np.float32)

train_data = Dataset(X, y)

# Train model
config = GBDTConfig(
    n_estimators=50,
    max_depth=5,
    learning_rate=0.1,
    objective=Objective.squared(),
    metric=Metric.rmse(),
)

print("Training model...")
model = GBDTModel.train(train_data, config=config)
print(f"Trained: {model}")

# Save with pickle
print("\nSaving model with pickle...")
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to model.pkl")

# Load with pickle
print("\nLoading model from pickle...")
with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

print(f"Loaded: {loaded_model}")

# Verify predictions match
original_preds = model.predict(train_data)
loaded_preds = loaded_model.predict(train_data)

print("\nVerifying predictions...")
np.testing.assert_array_almost_equal(original_preds, loaded_preds, decimal=6)
print("✓ Predictions match!")

# Pickle also works with pickle.dumps/loads for in-memory serialization
print("\nIn-memory pickle serialization...")
pickled_bytes = pickle.dumps(model)
print(f"Pickled size: {len(pickled_bytes):,} bytes")

restored = pickle.loads(pickled_bytes)
restored_preds = restored.predict(train_data)
np.testing.assert_array_almost_equal(original_preds, restored_preds, decimal=6)
print("✓ In-memory pickle works!")
