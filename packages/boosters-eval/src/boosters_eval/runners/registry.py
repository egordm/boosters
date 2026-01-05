"""Runner registry and dependency checks."""

from __future__ import annotations

from boosters_eval.runners.base import Runner
from boosters_eval.runners.boosters import BoostersRunner
from boosters_eval.runners.lightgbm import LightGBMRunner
from boosters_eval.runners.xgboost import XGBoostRunner

_RUNNERS: dict[str, type[Runner]] = {
    "boosters": BoostersRunner,
    "xgboost": XGBoostRunner,
    "lightgbm": LightGBMRunner,
}


def get_runner(name: str) -> type[Runner]:
    """Get a runner by name.

    Args:
        name: Runner name (boosters, xgboost, lightgbm).

    Returns:
        Runner class.

    Raises:
        KeyError: If runner not found.
        ImportError: If runner's library not installed.
    """
    if name not in _RUNNERS:
        raise KeyError(f"Unknown runner: {name}")

    runner_cls = _RUNNERS[name]

    if name == "xgboost":
        import xgboost  # noqa: F401
    elif name == "lightgbm":
        import lightgbm  # noqa: F401
    elif name == "boosters":
        import boosters  # noqa: F401

    return runner_cls


def get_available_runners() -> list[str]:
    """Get list of runners with available dependencies."""
    available: list[str] = []
    for name in _RUNNERS:
        try:
            get_runner(name)
            available.append(name)
        except ImportError:
            pass
    return available


__all__ = ["get_available_runners", "get_runner"]
