from .detect import Hardware, detect, resolve_profile
from .profiles import (
    TIERS,
    default_context,
    default_kv_bits,
    default_single_resident,
    default_vision,
    model_budget_gb,
    select_profile,
)

__all__ = [
    "Hardware",
    "detect",
    "resolve_profile",
    "TIERS",
    "default_context",
    "default_kv_bits",
    "default_single_resident",
    "default_vision",
    "model_budget_gb",
    "select_profile",
]
