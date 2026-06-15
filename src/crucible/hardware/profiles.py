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


def select_profile(total_gb: float) -> str:
    """Highest tier whose minimum total memory fits the detected machine.

    Below the smallest tier, fall back to the smallest tier rather than failing.
    """
    chosen = TIERS[0].name
    for tier in TIERS:
        if total_gb + 1e-6 >= tier.min_total_gb:
            chosen = tier.name
    return chosen
