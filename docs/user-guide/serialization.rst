=============
Serialization
=============

Save and load trained models for production deployment, reproducibility, 
and sharing.

Quick Start
-----------

**sklearn API:**

.. code-block:: python

   from boosters.sklearn import GBDTRegressor
   import joblib

   # Train model
   model = GBDTRegressor()
   model.fit(X_train, y_train)

   # Save with joblib (recommended for sklearn)
   joblib.dump(model, "model.joblib")

   # Load
   model = joblib.load("model.joblib")
   predictions = model.predict(X_test)

**Core API:**

.. code-block:: python

   import boosters as bst

   # Train model
   model = bst.GBDTModel.train(train_data, config=config)

   # Save to binary format (recommended)
   model.save("model.bst")

   # Load
   model = bst.GBDTModel.load("model.bst")
   predictions = model.predict(test_data)

File Formats
------------

.. list-table::
   :widths: 20 30 25 25
   :header-rows: 1

   * - Format
     - Extension
     - Best For
     - API
   * - **Boosters Binary**
     - ``.bst``
     - Core API, production
     - Core API only
   * - **Joblib**
     - ``.joblib``
     - sklearn integration
     - sklearn API
   * - **Pickle**
     - ``.pkl``
     - General Python
     - Both
   * - **JSON**
     - ``.json``
     - Debugging, inspection
     - Core API only

Boosters Binary Format (.bst)
-----------------------------

The native binary format is fast and compact:

.. code-block:: python

   import boosters as bst

   # Save
   model.save("model.bst")

   # Load
   model = bst.GBDTModel.load("model.bst")

**Features:**

- ✅ Fastest save/load
- ✅ Smallest file size
- ✅ Version-compatible (forward and backward)
- ❌ Core API only (not sklearn wrapper)

Joblib (sklearn API)
--------------------

Standard for sklearn models:

.. code-block:: python

   import joblib

   # Save
   joblib.dump(model, "model.joblib")

   # Load
   model = joblib.load("model.joblib")

**With compression:**

.. code-block:: python

   # Gzip compression (smaller files)
   joblib.dump(model, "model.joblib.gz", compress=3)

   # Load automatically decompresses
   model = joblib.load("model.joblib.gz")

JSON Export (Inspection)
------------------------

Export to human-readable JSON for debugging:

.. code-block:: python

   # Export to JSON
   model.save_json("model.json")

   # Load from JSON
   model = bst.GBDTModel.load_json("model.json")

**Example JSON structure:**

.. code-block:: json

   {
     "version": "0.1.0",
     "model_type": "gbdt",
     "config": {
       "n_estimators": 100,
       "max_depth": 6,
       "learning_rate": 0.1
     },
     "trees": [
       {
         "nodes": [
           {"feature": 0, "threshold": 0.5, "left": 1, "right": 2},
           {"leaf_value": 0.123},
           {"leaf_value": -0.456}
         ]
       }
     ]
   }

Version Compatibility
---------------------

Boosters maintains backward compatibility:

.. code-block:: python

   # Old model trained with boosters 0.1.0
   model = bst.GBDTModel.load("old_model.bst")

   # Works with boosters 0.2.0+
   predictions = model.predict(X)

**Forward compatibility:**

When loading a model from a newer version, only supported features are used.
A warning is issued if unsupported features are present.

Production Deployment
---------------------

**Docker deployment:**

.. code-block:: dockerfile

   FROM python:3.11-slim

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY model.bst /app/model.bst
   COPY serve.py /app/serve.py

   CMD ["python", "/app/serve.py"]

**FastAPI example:**

.. code-block:: python

   from fastapi import FastAPI
   import boosters as bst
   import numpy as np

   app = FastAPI()

   # Load model once at startup
   model = bst.GBDTModel.load("model.bst")

   @app.post("/predict")
   async def predict(features: list[float]):
       X = np.array([features])
       prediction = model.predict(X)
       return {"prediction": float(prediction[0])}

**Flask example:**

.. code-block:: python

   from flask import Flask, request, jsonify
   import boosters as bst
   import numpy as np

   app = Flask(__name__)
   model = bst.GBDTModel.load("model.bst")

   @app.route("/predict", methods=["POST"])
   def predict():
       features = request.json["features"]
       X = np.array([features])
       prediction = model.predict(X)
       return jsonify({"prediction": float(prediction[0])})

Model Artifacts
---------------

For reproducibility, save more than just the model:

.. code-block:: python

   import json
   import joblib

   def save_model_artifacts(model, path, feature_names, metadata):
       """Save model with all artifacts for reproducibility."""
       artifacts = {
           "feature_names": feature_names,
           "metadata": metadata,  # training date, version, etc.
           "n_features": len(feature_names),
       }
       
       # Save artifacts
       with open(f"{path}/artifacts.json", "w") as f:
           json.dump(artifacts, f, indent=2)
       
       # Save model
       model.save(f"{path}/model.bst")

   def load_model_artifacts(path):
       """Load model and artifacts."""
       with open(f"{path}/artifacts.json") as f:
           artifacts = json.load(f)
       
       model = bst.GBDTModel.load(f"{path}/model.bst")
       return model, artifacts

Security Considerations
-----------------------

.. warning::

   Never load models from untrusted sources. Pickle and joblib files can 
   execute arbitrary code when loaded.

**Safe loading:**

.. code-block:: python

   # Verify checksum before loading
   import hashlib

   def load_verified(path, expected_hash):
       with open(path, "rb") as f:
           actual_hash = hashlib.sha256(f.read()).hexdigest()
       
       if actual_hash != expected_hash:
           raise ValueError("Model file hash mismatch!")
       
       return bst.GBDTModel.load(path)

See Also
--------

- :doc:`production` — Production deployment patterns
- :doc:`/tutorials/09-model-serialization` — Hands-on serialization tutorial
