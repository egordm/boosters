==============
Explainability
==============

Understanding *why* your model makes predictions is critical for debugging, 
trust, and regulatory compliance. Boosters provides multiple explainability methods.

Quick Start
-----------

.. code-block:: python

   from boosters.sklearn import GBDTRegressor

   model = GBDTRegressor()
   model.fit(X_train, y_train)

   # Global feature importance
   importance = model.feature_importances_

   # Local explanations (per-prediction)
   contributions = model.predict_contributions(X_test)

Feature Importance Methods
--------------------------

**Split-based importance** (how often a feature is used):

.. code-block:: python

   # sklearn API
   importance = model.feature_importances_  # Uses 'split' by default

   # Core API
   importance = model.feature_importance(importance_type="split")

**Gain-based importance** (how much a feature improves the loss):

.. code-block:: python

   # sklearn API (specify in constructor)
   model = GBDTRegressor(importance_type="gain")
   model.fit(X, y)
   importance = model.feature_importances_

   # Core API
   importance = model.feature_importance(importance_type="gain")

**Comparison:**

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Method
     - Measures
     - Best For
   * - **split**
     - Number of times feature is used
     - Feature selection, understanding model structure
   * - **gain**
     - Total loss reduction from splits
     - Understanding predictive power

Visualizing Importance
----------------------

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt

   # Create importance DataFrame
   importance_df = pd.DataFrame({
       "feature": feature_names,
       "importance": model.feature_importances_,
   }).sort_values("importance", ascending=True)

   # Plot
   plt.figure(figsize=(10, 6))
   plt.barh(importance_df["feature"], importance_df["importance"])
   plt.xlabel("Importance")
   plt.title("Feature Importance")
   plt.tight_layout()
   plt.show()

Local Explanations (SHAP-style)
-------------------------------

Feature contributions explain individual predictions by showing how each 
feature pushed the prediction away from the baseline (average prediction).

.. code-block:: python

   # Get contributions for each sample
   contributions = model.predict_contributions(X_test)
   # Shape: (n_samples, n_features + 1)
   # Last column is the bias (base value)

   # For a single prediction:
   sample_idx = 0
   for i, (name, contrib) in enumerate(zip(feature_names, contributions[sample_idx])):
       print(f"{name}: {contrib:+.3f}")
   print(f"Bias: {contributions[sample_idx, -1]:.3f}")
   print(f"Sum = Prediction: {contributions[sample_idx].sum():.3f}")

**Output example:**

::

   temperature: +2.341
   humidity: -0.892
   pressure: +0.156
   wind_speed: -0.234
   Bias: 15.230
   Sum = Prediction: 16.601

Understanding Contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Positive values**: Feature pushes prediction higher
- **Negative values**: Feature pushes prediction lower
- **Bias**: The baseline prediction (mean of training targets)
- **Sum**: Contributions + bias = prediction

Waterfall Visualization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   def plot_waterfall(contributions, feature_names, prediction):
       """Simple waterfall plot for a single prediction."""
       bias = contributions[-1]
       contribs = contributions[:-1]
       
       # Sort by absolute contribution
       sorted_idx = np.argsort(np.abs(contribs))[::-1]
       
       fig, ax = plt.subplots(figsize=(10, 6))
       
       cumsum = bias
       for i, idx in enumerate(sorted_idx):
           color = 'green' if contribs[idx] > 0 else 'red'
           ax.barh(i, contribs[idx], left=cumsum if contribs[idx] > 0 else cumsum + contribs[idx], color=color)
           cumsum += contribs[idx]
       
       ax.set_yticks(range(len(sorted_idx)))
       ax.set_yticklabels([feature_names[i] for i in sorted_idx])
       ax.axvline(x=bias, color='gray', linestyle='--', label=f'Bias: {bias:.2f}')
       ax.axvline(x=prediction, color='blue', linestyle='-', label=f'Prediction: {prediction:.2f}')
       ax.legend()
       plt.tight_layout()
       plt.show()

   # Usage
   sample = 0
   plot_waterfall(
       contributions[sample], 
       feature_names, 
       model.predict(X_test[sample:sample+1])[0]
   )

SHAP Integration
----------------

For advanced visualizations, use the SHAP library:

.. code-block:: python

   import shap

   # TreeExplainer works with boosters models
   explainer = shap.TreeExplainer(model)
   shap_values = explainer.shap_values(X_test)

   # Summary plot
   shap.summary_plot(shap_values, X_test, feature_names=feature_names)

   # Dependence plot
   shap.dependence_plot("feature_name", shap_values, X_test)

   # Force plot (single prediction)
   shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

Partial Dependence
------------------

Show how predictions change as a feature varies:

.. code-block:: python

   from sklearn.inspection import PartialDependenceDisplay

   # 1D partial dependence
   PartialDependenceDisplay.from_estimator(
       model, X_train, features=[0, 1],  # Feature indices
       feature_names=feature_names,
   )
   plt.show()

   # 2D interaction plot
   PartialDependenceDisplay.from_estimator(
       model, X_train, features=[(0, 1)],  # Feature pair
       feature_names=feature_names,
   )
   plt.show()

Tree Visualization
------------------

Inspect individual trees:

.. code-block:: python

   # Get tree structure
   tree_info = model.get_tree(tree_idx=0)

   # Export to graphviz (if available)
   model.export_tree_graphviz(tree_idx=0, filename="tree_0.dot")

Best Practices
--------------

1. **Use gain for predictive importance, split for model structure**

2. **Always check multiple samples** — Contributions vary per sample

3. **Validate with domain knowledge** — If important features don't make sense, investigate

4. **Consider feature correlations** — Correlated features can "share" importance

5. **Use SHAP for stakeholder communication** — Better visualizations

See Also
--------

- :doc:`/tutorials/08-explainability` — Full tutorial with examples
- :doc:`/research/explainability` — Research background on explainability methods
