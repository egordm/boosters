=====================
Production Deployment
=====================

This guide covers best practices for deploying boosters models in production 
environments.

Model Serialization
-------------------

Save and load models for deployment:

Pickle (Recommended)
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import pickle
   from boosters.sklearn import GBDTRegressor

   # Train
   model = GBDTRegressor(n_estimators=100)
   model.fit(X_train, y_train)

   # Save
   with open("model.pkl", "wb") as f:
       pickle.dump(model, f)

   # Load (in production)
   with open("model.pkl", "rb") as f:
       model = pickle.load(f)
   
   predictions = model.predict(X_new)

Joblib (Large Models)
^^^^^^^^^^^^^^^^^^^^^

For larger models, joblib provides better compression:

.. code-block:: python

   import joblib

   # Save with compression
   joblib.dump(model, "model.joblib", compress=3)

   # Load
   model = joblib.load("model.joblib")

Inference Optimization
----------------------

Batch Processing
^^^^^^^^^^^^^^^^

Always batch predictions when possible:

.. code-block:: python

   # ❌ Slow
   predictions = [model.predict(x.reshape(1, -1))[0] for x in X]

   # ✅ Fast
   predictions = model.predict(X)

Pre-allocate Output
^^^^^^^^^^^^^^^^^^^

For repeated predictions, pre-allocate arrays:

.. code-block:: python

   import numpy as np

   # Pre-allocate
   buffer = np.empty(batch_size, dtype=np.float32)

   for batch in batches:
       model.predict(batch, out=buffer)  # If supported
       process(buffer)

Warm Starts
^^^^^^^^^^^

If predicting repeatedly, the model is already in cache:

.. code-block:: python

   # First prediction may be slower (loading model into cache)
   _ = model.predict(X_sample[:1])

   # Subsequent predictions are faster
   predictions = model.predict(X_production)

API Deployment
--------------

FastAPI Example
^^^^^^^^^^^^^^^

.. code-block:: python

   from fastapi import FastAPI
   import numpy as np
   import pickle

   app = FastAPI()

   # Load model once at startup
   with open("model.pkl", "rb") as f:
       model = pickle.load(f)

   @app.post("/predict")
   async def predict(features: list[float]):
       X = np.array([features])
       prediction = model.predict(X)[0]
       return {"prediction": float(prediction)}

Model Versioning
^^^^^^^^^^^^^^^^

Track model versions for reproducibility:

.. code-block:: python

   import hashlib
   import json

   def model_signature(model, X_sample):
       """Create a signature for model verification."""
       predictions = model.predict(X_sample[:10])
       return hashlib.md5(predictions.tobytes()).hexdigest()

   # Save with metadata
   metadata = {
       "version": "1.0.0",
       "created": "2024-01-15",
       "signature": model_signature(model, X_train),
       "n_features": X_train.shape[1],
   }
   
   with open("model_metadata.json", "w") as f:
       json.dump(metadata, f)

Input Validation
----------------

Validate inputs before prediction:

.. code-block:: python

   import numpy as np

   def validate_input(X, expected_features):
       """Validate input array."""
       X = np.asarray(X)
       
       if X.ndim == 1:
           X = X.reshape(1, -1)
       
       if X.shape[1] != expected_features:
           raise ValueError(
               f"Expected {expected_features} features, got {X.shape[1]}"
           )
       
       if np.isnan(X).all(axis=0).any():
           raise ValueError("Some features are entirely NaN")
       
       return X

   # Usage
   X_validated = validate_input(user_input, n_features=50)
   prediction = model.predict(X_validated)

Monitoring
----------

Log predictions for monitoring:

.. code-block:: python

   import logging
   import time

   logger = logging.getLogger("model")

   def predict_with_logging(model, X):
       start = time.perf_counter()
       predictions = model.predict(X)
       elapsed = time.perf_counter() - start
       
       logger.info(
           "Prediction completed",
           extra={
               "batch_size": len(X),
               "latency_ms": elapsed * 1000,
               "predictions_mean": float(predictions.mean()),
               "predictions_std": float(predictions.std()),
           }
       )
       
       return predictions

A/B Testing
-----------

Compare model versions:

.. code-block:: python

   import random

   class ModelRouter:
       def __init__(self, model_a, model_b, traffic_split=0.5):
           self.model_a = model_a
           self.model_b = model_b
           self.traffic_split = traffic_split
       
       def predict(self, X, experiment_id=None):
           if random.random() < self.traffic_split:
               model = self.model_a
               variant = "A"
           else:
               model = self.model_b
               variant = "B"
           
           predictions = model.predict(X)
           
           # Log for analysis
           logger.info(f"Variant {variant}", extra={"experiment_id": experiment_id})
           
           return predictions, variant

Performance Checklist
---------------------

Before deploying:

☐ Model file size is acceptable for your infrastructure
☐ Inference latency meets requirements (test with production batch sizes)
☐ Memory usage is within limits
☐ Input validation is in place
☐ Logging and monitoring are configured
☐ Model versioning and rollback plan exists
☐ A/B testing framework is ready (if applicable)

See Also
--------

- :doc:`debugging-performance` — Optimizing model performance
- :doc:`/explanations/benchmarks` — Performance characteristics
