//! Python bindings for the boosters library.
//!
//! This crate provides PyO3-based bindings for training and inference
//! with gradient boosted models.
//!
//! # sklearn-style API
//!
//! Models follow the sklearn convention:
//! - Constructor takes hyperparameters
//! - `fit(X, y)` takes training data
//! - `predict(X)` makes predictions
//!
//! ```python
//! from boosters_python import GBDTBooster
//!
//! # Create model with hyperparameters
//! model = GBDTBooster(n_estimators=100, learning_rate=0.1, max_depth=6)
//!
//! # Fit on training data
//! model.fit(X_train, y_train)
//!
//! # Make predictions
//! predictions = model.predict(X_test)
//! ```

use pyo3::prelude::*;

mod error;
mod gbdt;
mod linear;
mod params;

/// Get the version of the boosters library.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Python module for boosters gradient boosting library.
#[pymodule]
#[pyo3(name = "_boosters_python")]
fn boosters_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(version, m)?)?;

    // Models (sklearn-style: constructor takes params, fit takes data)
    m.add_class::<gbdt::PyGBDTBooster>()?;
    m.add_class::<linear::PyGBLinearBooster>()?;

    Ok(())
}
