"""Environment settings for the classify-goblin local runtime.

New deployments use ``CLASSIFY_GOBLIN_*`` names. The resolver accepts the former
``JEV_*`` spelling as a deliberately narrow migration fallback so an already
running local service can be upgraded without changing its secret store or
process supervisor at the same time. New names always win when both are set.
"""

from __future__ import annotations

import os
from typing import overload


_PREFIX = "CLASSIFY_GOBLIN_"
_LEGACY_PREFIX = "JEV_"


@overload
def get_env(name: str, default: str) -> str: ...


@overload
def get_env(name: str, default: None = None) -> str | None: ...


@overload
def get_env(name: str, default: str | None) -> str | None: ...


def get_env(name: str, default: str | None = None) -> str | None:
    """Resolve a classify-goblin setting, preferring its new name.

    Only names in the ``CLASSIFY_GOBLIN_`` namespace receive the compatibility
    fallback. Generic variables such as ``TYPESAFE_API_KEY`` remain explicit
    protocol compatibility settings and are not rewritten.
    """

    value = os.environ.get(name)
    if value is not None:
        return value
    if name.startswith(_PREFIX):
        return os.environ.get(_LEGACY_PREFIX + name[len(_PREFIX):], default)
    return default


__all__ = ["get_env"]
