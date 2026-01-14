=======
Recipes
=======

Quick solutions to common tasks.

Cross-Validation
----------------

.. code-block:: python

   from sklearn.model_selection import cross_val_score
   from boosters.sklearn import GBDTRegressor

   model = GBDTRegressor(n_estimators=100)
   scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
   print(f"RMSE: {-scores.mean():.4f} ± {scores.std():.4f}")

Hyperparameter Search
---------------------

.. code-block:: python

   from sklearn.model_selection import RandomizedSearchCV
   from scipy.stats import uniform, randint
   from boosters.sklearn import GBDTRegressor

   param_dist = {
       "n_estimators": randint(50, 200),
       "max_depth": randint(3, 8),
       "learning_rate": uniform(0.01, 0.3),
       "subsample": uniform(0.6, 0.4),
   }

   search = RandomizedSearchCV(
       GBDTRegressor(),
       param_dist,
       n_iter=20,
       cv=3,
       scoring="neg_root_mean_squared_error",
       n_jobs=-1,
   )
   search.fit(X, y)
   print(f"Best params: {search.best_params_}")

Feature Importance
------------------

.. code-block:: python

   from boosters.sklearn import GBDTClassifier
   import matplotlib.pyplot as plt

   model = GBDTClassifier()
   model.fit(X_train, y_train)

   # Get feature importance
   importance = model.feature_importances_
   
   # Plot
   plt.barh(range(len(importance)), importance)
   plt.xlabel("Importance")
   plt.ylabel("Feature")
   plt.tight_layout()
   plt.show()

Pipeline with Preprocessing
---------------------------

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import OneHotEncoder
   from boosters.sklearn import GBDTClassifier

   preprocessor = ColumnTransformer([
       ("num", StandardScaler(), numeric_features),
       ("cat", OneHotEncoder(drop="first"), categorical_features),
   ])

   pipeline = Pipeline([
       ("preprocess", preprocessor),
       ("model", GBDTClassifier(n_estimators=100)),
   ])

   pipeline.fit(X_train, y_train)

Save and Load with Pipeline
---------------------------

.. code-block:: python

   import joblib

   # Save entire pipeline
   joblib.dump(pipeline, "pipeline.joblib")

   # Load
   pipeline = joblib.load("pipeline.joblib")
   predictions = pipeline.predict(X_new)

Multi-Output Regression
-----------------------

.. code-block:: python

   from sklearn.multioutput import MultiOutputRegressor
   from boosters.sklearn import GBDTRegressor

   model = MultiOutputRegressor(GBDTRegressor(n_estimators=100))
   model.fit(X_train, y_train_multi)  # y has multiple columns
   predictions = model.predict(X_test)

Class Imbalance
---------------

.. code-block:: python

   from sklearn.utils.class_weight import compute_sample_weight
   from boosters.sklearn import GBDTClassifier

   # Compute sample weights
   weights = compute_sample_weight("balanced", y_train)

   model = GBDTClassifier(n_estimators=100)
   model.fit(X_train, y_train, sample_weight=weights)

Time Series Split
-----------------

.. code-block:: python

   from sklearn.model_selection import TimeSeriesSplit
   from boosters.sklearn import GBDTRegressor

   tscv = TimeSeriesSplit(n_splits=5)
   scores = []

   for train_idx, val_idx in tscv.split(X):
       model = GBDTRegressor(n_estimators=100)
       model.fit(X[train_idx], y[train_idx])
       score = model.score(X[val_idx], y[val_idx])
       scores.append(score)

   print(f"Mean R²: {np.mean(scores):.4f}")

Probability Calibration
-----------------------

.. code-block:: python

   from sklearn.calibration import CalibratedClassifierCV
   from boosters.sklearn import GBDTClassifier

   model = GBDTClassifier(n_estimators=100)
   calibrated = CalibratedClassifierCV(model, cv=5, method="isotonic")
   calibrated.fit(X_train, y_train)

   probabilities = calibrated.predict_proba(X_test)

Model Comparison
----------------

.. code-block:: python

   from sklearn.model_selection import cross_validate
   from boosters.sklearn import GBDTRegressor, GBLinearRegressor

   models = {
       "GBDT": GBDTRegressor(n_estimators=100),
       "GBLinear": GBLinearRegressor(n_estimators=100),
   }

   for name, model in models.items():
       cv_results = cross_validate(
           model, X, y, cv=5,
           scoring="neg_root_mean_squared_error",
           return_train_score=True,
       )
       print(f"{name}:")
       print(f"  Train RMSE: {-cv_results['train_score'].mean():.4f}")
       print(f"  Test RMSE:  {-cv_results['test_score'].mean():.4f}")
