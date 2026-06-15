"""Resolve the active runtime profile: the limits that govern the model manager.

The memory ceiling and single-vs-multi-resident behavior come from the active hardware
profile (with an explicit config override), so the manager runs unchanged across tiers.
"""

from __future__ import annotations

from dataclasses import dataclass

from crucible.config import Registry
from crucible.hardware import (
    default_context,
    default_kv_bits,
    default_single_resident,
    default_vision,
)

_GIB = 1024**3


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    ceiling_bytes: int
    single_resident: bool
    default_context: int
    kv_bits: int
    vision: bool

    @property
    def ceiling_gb(self) -> float:
        return round(self.ceiling_bytes / _GIB, 1)


def resolve_runtime(registry: Registry, active: str, budget_gb: float) -> RuntimeProfile:
    """Combine config overrides with tier defaults for the active profile.

    The ceiling is the explicit server.memory_ceiling_gb when set, else the detected
    hardware model budget.
    """
    ceiling_gb = registry.server.memory_ceiling_gb or budget_gb
    spec = registry.profiles.get(active)
    if spec is not None:
        return RuntimeProfile(
            name=active,
            ceiling_bytes=int(ceiling_gb * _GIB),
            single_resident=spec.single_resident,
            default_context=spec.default_context,
            kv_bits=spec.kv_bits,
            vision=spec.vision,
        )
    return RuntimeProfile(
        name=active,
        ceiling_bytes=int(ceiling_gb * _GIB),
        single_resident=default_single_resident(active),
        default_context=default_context(active),
        kv_bits=default_kv_bits(active),
        vision=default_vision(active),
    )
