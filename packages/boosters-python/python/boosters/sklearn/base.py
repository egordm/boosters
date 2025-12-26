"""Base classes and utilities for sklearn integration.

This module provides shared logic for sklearn-compatible estimators,
including base classes and validation utilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# =============================================================================
# sklearn Compatibility Layer
# =============================================================================

# Check if sklearn is available
try:
    from sklearn.base import BaseEstimator as _SklearnBaseEstimator
    from sklearn.base import ClassifierMixin as _SklearnClassifierMixin
    from sklearn.base import RegressorMixin as _SklearnRegressorMixin
    from sklearn.utils.validation import (
        check_array as _sklearn_check_array,
    )
    from sklearn.utils.validation import (
        check_is_fitted as _sklearn_check_is_fitted,
    )
    from sklearn.utils.validation import (
        check_X_y as _sklearn_check_X_y,
    )

    SKLEARN_AVAILABLE = True

    # Use sklearn implementations
    BaseEstimator: type = _SklearnBaseEstimator
    ClassifierMixin: type = _SklearnClassifierMixin
    RegressorMixin: type = _SklearnRegressorMixin

    def check_array(X: Any, **kwargs: Any) -> NDArray[np.floating[Any]]:
        """Validate array input."""
        return _sklearn_check_array(X, **kwargs)  # type: ignore[return-value]

    def check_X_y(X: Any, y: Any, **kwargs: Any) -> tuple[NDArray[np.floating[Any]], NDArray[Any]]:
        """Validate X and y inputs."""
        return _sklearn_check_X_y(X, y, **kwargs)  # type: ignore[return-value]

    def check_is_fitted(estimator: Any, attributes: list[str] | None = None) -> None:
        """Check if estimator is fitted."""
        _sklearn_check_is_fitted(estimator, attributes)

except ImportError:
    # Create dummy classes if sklearn is not available
    SKLEARN_AVAILABLE = False

    class _DummyBaseEstimator:
        """Dummy base class when sklearn is not installed.

        Provides minimal get_params/set_params for sklearn compatibility.
        """

        def get_params(self, deep: bool = True) -> dict[str, Any]:
            """Get parameters for this estimator."""
            params = {}
            for key in self.__init__.__code__.co_varnames[1:]:  # Skip 'self'
                if hasattr(self, key):
                    params[key] = getattr(self, key)
            return params

        def set_params(self, **params: Any) -> _DummyBaseEstimator:
            """Set parameters for this estimator."""
            for key, value in params.items():
                setattr(self, key, value)
            return self

    class _DummyRegressorMixin:
        """Dummy mixin when sklearn is not installed."""

        pass

    class _DummyClassifierMixin:
        """Dummy mixin when sklearn is not installed."""

        pass

    BaseEstimator: type = _DummyBaseEstimator  # type: ignore[no-redef]
    ClassifierMixin: type = _DummyClassifierMixin  # type: ignore[no-redef]
    RegressorMixin: type = _DummyRegressorMixin  # type: ignore[no-redef]

    def check_array(X: Any, **kwargs: Any) -> NDArray[np.floating[Any]]:  # noqa: D103
        return np.asarray(X, dtype=np.float32)

    def check_X_y(  # noqa: D103
        X: Any, y: Any, **kwargs: Any
    ) -> tuple[NDArray[np.floating[Any]], NDArray[Any]]:
        # Don't convert y to float32 since it might be categorical
        return np.asarray(X, dtype=np.float32), np.asarray(y)

    def check_is_fitted(  # noqa: D103
        estimator: Any, attributes: list[str] | None = None
    ) -> None:
        pass


__all__ = [
    "SKLEARN_AVAILABLE",
    "BaseEstimator",
    "ClassifierMixin",
    "RegressorMixin",
    "check_X_y",
    "check_array",
    "check_is_fitted",
]
