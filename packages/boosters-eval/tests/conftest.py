"""Pytest configuration for boosters-eval tests."""

from __future__ import annotations

import pytest


def _check_xgboost() -> bool:
    """Check if xgboost is available."""
    try:
        import xgboost  # noqa: F401

        return True
    except ImportError:
        return False


def _check_lightgbm() -> bool:
    """Check if lightgbm is available."""
    try:
        import lightgbm  # noqa: F401

        return True
    except ImportError:
        return False


# Cache availability checks
_XGBOOST_AVAILABLE = _check_xgboost()
_LIGHTGBM_AVAILABLE = _check_lightgbm()


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "xgboost: tests requiring xgboost")
    config.addinivalue_line("markers", "lightgbm: tests requiring lightgbm")


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip tests based on marker and available dependencies."""
    skip_xgboost = pytest.mark.skip(reason="xgboost not installed")
    skip_lightgbm = pytest.mark.skip(reason="lightgbm not installed")

    for item in items:
        if "xgboost" in item.keywords and not _XGBOOST_AVAILABLE:
            item.add_marker(skip_xgboost)
        if "lightgbm" in item.keywords and not _LIGHTGBM_AVAILABLE:
            item.add_marker(skip_lightgbm)


@pytest.fixture
def xgboost_available() -> bool:
    """Check if xgboost is available."""
    return _XGBOOST_AVAILABLE


@pytest.fixture
def lightgbm_available() -> bool:
    """Check if lightgbm is available."""
    return _LIGHTGBM_AVAILABLE
