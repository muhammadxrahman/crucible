"""Hardware profile tiers and selection.

A profile bundles the limits that change per memory tier. Auto-detection picks the
highest tier whose minimum total memory fits the detected machine. See docs/models.md
for the tier table and docs/hardware.md for the budgeting rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

# macOS reserve and process working set, in GiB (docs/hardware.md).
_OS_RESERVE_GB = 4.0


@dataclass(frozen=True)
class ProfileTier:
    name: str
    min_total_gb: int


# Ordered smallest to largest.
TIERS: tuple[ProfileTier, ...] = (
    ProfileTier("air16", 16),
    ProfileTier("base24", 24),
    ProfileTier("pro32", 32),
    ProfileTier("pro48", 48),
    ProfileTier("pro64", 64),
    ProfileTier("max128", 128),
)


def working_set_gb(total_gb: float) -> float:
    """Reserve for IDE, browser, other apps, and the server process."""
    return 8.0 if total_gb <= 24 else 12.0


def model_budget_gb(total_gb: float) -> float:
    """Total unified memory minus the macOS reserve and the working set."""
    return max(0.0, total_gb - _OS_RESERVE_GB - working_set_gb(total_gb))


# Tiers that default to a single resident model (small memory). Larger tiers default
# to multi-resident. A config profile entry overrides these defaults (docs/models.md).
_SINGLE_RESIDENT_TIERS = frozenset({"air16", "base24", "pro32"})
_NO_VISION_TIERS = frozenset({"air16"})


def default_single_resident(profile: str) -> bool:
    return profile in _SINGLE_RESIDENT_TIERS


def default_vision(profile: str) -> bool:
    return profile not in _NO_VISION_TIERS


def default_context(profile: str) -> int:
    return 8192 if profile in _SINGLE_RESIDENT_TIERS else 32768


def default_kv_bits(profile: str) -> int:
    return 4 if profile in _SINGLE_RESIDENT_TIERS else 8


def select_profile(total_gb: float) -> str:
    """Highest tier whose minimum total memory fits the detected machine.

    Below the smallest tier, fall back to the smallest tier rather than failing.
    """
    chosen = TIERS[0].name
    for tier in TIERS:
        if total_gb + 1e-6 >= tier.min_total_gb:
            chosen = tier.name
    return chosen
