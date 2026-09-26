"""Registry of optimisers, discovered automatically from this package's modules."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

import numpy as np

PREFERRED_ORDER = (
    "standard_de",
    "jade",
    "shade",
    "lshade",
    "late_acceptance_de",
)

Optimizer = Callable[..., tuple[np.ndarray, float, np.ndarray, np.ndarray]]


def _public_modules() -> Iterator[tuple[str, ModuleType]]:
    for _, module_name, _ in pkgutil.iter_modules(__path__):
        if not module_name.startswith("_"):
            yield module_name, importlib.import_module(f"{__name__}.{module_name}")


def _entry_point(module_name: str, module: ModuleType) -> Optimizer | None:
    """The function named after the module, else its single optimiser class."""
    function = getattr(module, module_name, None)
    if callable(function) and not inspect.isclass(function):
        return function

    classes = _optimizer_classes(module)
    if len(classes) != 1:
        return None
    return _instantiating(classes[0])


def _optimizer_classes(module: ModuleType) -> list[type]:
    return [
        obj
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and _is_optimizer_class(obj)
        and obj.__module__ == module.__name__
    ]


def _is_optimizer_class(obj: Any) -> bool:
    return (
        inspect.isclass(obj)
        and hasattr(obj, "optimize")
        and callable(obj.optimize)
    )


def _instantiating(optimizer_class: type) -> Optimizer:
    return lambda *args, **kwargs: optimizer_class().optimize(*args, **kwargs)


def _discover() -> dict[str, Optimizer]:
    entries = {name: _entry_point(name, module) for name, module in _public_modules()}
    return {name: entry for name, entry in entries.items() if entry is not None}


ALGORITHMS: dict[str, Optimizer] = _discover()


def available() -> list[str]:
    """Report order first, then any extras alphabetically."""
    known = [name for name in PREFERRED_ORDER if name in ALGORITHMS]
    extra = sorted(set(ALGORITHMS) - set(PREFERRED_ORDER))
    return known + extra


def get(name: str) -> Optimizer:
    try:
        return ALGORITHMS[name]
    except KeyError:
        raise KeyError(
            f"unknown algorithm {name!r}; available: {available()}"
        ) from None


__all__ = ["ALGORITHMS", "PREFERRED_ORDER", "available", "get"]
