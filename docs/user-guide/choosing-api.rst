==================
Choosing an API
==================

boosters provides **two Python APIs** designed for different use cases:

.. list-table::
   :widths: 15 40 45
   :header-rows: 1

   * - API
     - Best For
     - Key Features
   * - **sklearn API**
     - Most users, ML pipelines
     - Works with ``Pipeline``, ``cross_val_score``, ``GridSearchCV``
   * - **Core API**
     - Full control, custom workflows
     - Explicit ``Dataset``, callbacks, advanced configuration

sklearn API
-----------

**Use the sklearn API when you want:**

- Integration with scikit-learn pipelines and tools
- Familiar estimator interface (``fit``/``predict``/``score``)
- Quick experimentation and prototyping
- Hyperparameter tuning with ``GridSearchCV`` or ``RandomizedSearchCV``

**Example:**

.. code-block:: python

   from boosters.sklearn import GBDTRegressor, GBDTClassifier
   from sklearn.model_selection import cross_val_score
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler

   # Simple usage
   model = GBDTRegressor(n_estimators=100, max_depth=6)
   model.fit(X_train, y_train)
   predictions = model.predict(X_test)

   # With sklearn tools
   scores = cross_val_score(model, X, y, cv=5)

   # In a pipeline
   pipe = Pipeline([
       ('scaler', StandardScaler()),
       ('model', GBDTRegressor(n_estimators=100)),
   ])
   pipe.fit(X_train, y_train)

**Available estimators:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Use Case
   * - ``GBDTRegressor``
     - Regression with decision trees
   * - ``GBDTClassifier``
     - Classification with decision trees
   * - ``GBLinearRegressor``
     - Regression with linear boosting (sparse data)
   * - ``GBLinearClassifier``
     - Classification with linear boosting

Core API
--------

**Use the Core API when you want:**

- Full control over the training process
- Custom objectives or metrics
- Access to training callbacks and logging
- Advanced configuration options
- Direct control over the ``Dataset`` object

**Example:**

.. code-block:: python

   import boosters as bst

   # Create dataset explicitly
   train_data = bst.Dataset(X_train, y_train)
   val_data = bst.Dataset(X_val, y_val)

   # Configure with all options
   config = bst.GBDTConfig(
       n_estimators=100,
       learning_rate=0.1,
       max_depth=6,
       objective=bst.Objective.squared(),
       metric=bst.Metric.rmse(),
       early_stopping_rounds=10,
       l2=1.0,
       subsample=0.8,
       colsample_bytree=0.8,
   )

   # Train with validation set
   model = bst.GBDTModel.train(
       train_data,
       evals=[(val_data, "val")],
       config=config,
   )

   # Predict
   predictions = model.predict(bst.Dataset(X_test))

**Core API components:**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Description
   * - ``Dataset``
     - Wraps feature matrix and optional labels/weights
   * - ``GBDTModel``
     - Gradient boosted decision trees
   * - ``GBLinearModel``
     - Gradient boosted linear model
   * - ``GBDTConfig``
     - Configuration for GBDT training
   * - ``GBLinearConfig``
     - Configuration for GBLinear training
   * - ``Objective``
     - Loss functions (squared, logistic, softmax, etc.)
   * - ``Metric``
     - Evaluation metrics (RMSE, AUC, accuracy, etc.)

Key Differences
---------------

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Feature
     - sklearn API
     - Core API
   * - **Interface**
     - ``fit(X, y)`` / ``predict(X)``
     - ``train(dataset)`` / ``predict(dataset)``
   * - **Input format**
     - NumPy arrays directly
     - Explicit ``Dataset`` objects
   * - **Configuration**
     - Constructor kwargs
     - ``GBDTConfig`` / ``GBLinearConfig``
   * - **Validation**
     - Use sklearn's ``cross_val_score``
     - Pass ``evals`` list to ``train()``
   * - **sklearn tools**
     - ✅ Full support
     - ❌ Not directly compatible
   * - **Callbacks**
     - ❌ Not available
     - ✅ Available
   * - **Custom objectives**
     - Via ``objective=`` param
     - Via ``config.objective``

When to Use Each
----------------

**Choose sklearn API if:**

- You're already using scikit-learn
- You want to use ``Pipeline``, ``GridSearchCV``, etc.
- You're doing quick experiments
- You don't need advanced control

**Choose Core API if:**

- You need maximum control over training
- You're implementing custom training loops
- You want to use callbacks or custom logging
- You're optimizing for production performance

Both APIs produce the same models — the difference is in how you interact with them.
