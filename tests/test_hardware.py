"""M0 acceptance: profile selection and memory budgeting.

Bandwidth, not chip generation, predicts decode speed, but profile selection is keyed
to total memory (docs/hardware.md, docs/models.md).
"""

import pytest

from crucible.hardware import detect, model_budget_gb, resolve_profile, select_profile


@pytest.mark.parametrize(
    "total_gb,expected",
    [
        (8, "air16"),  # below smallest tier falls back to smallest
        (16, "air16"),
        (18, "air16"),
        (24, "base24"),
        (32, "pro32"),
        (36, "pro32"),
        (48, "pro48"),
        (64, "pro64"),
        (96, "pro64"),
        (128, "max128"),
        (192, "max128"),
    ],
)
def test_select_profile_tiers(total_gb: float, expected: str) -> None:
    assert select_profile(total_gb) == expected


def test_budget_leaves_headroom() -> None:
    # 64GB target: ~48GB budget after macOS reserve and working set.
    assert 46 <= model_budget_gb(64) <= 50
    assert model_budget_gb(16) < 16
    assert model_budget_gb(8) >= 0  # never negative


def test_resolve_profile_explicit_overrides_auto() -> None:
    assert resolve_profile("auto", 64) == "pro64"
    assert resolve_profile("air16", 64) == "air16"  # explicit wins


def test_detect_this_machine_is_64gb_pro64() -> None:
    # This dev machine is a real M5 Pro / 64GB; M0 acceptance requires correct detection.
    hw = detect()
    assert hw.total_gb == pytest.approx(64, abs=1)
    assert hw.profile == "pro64"
    assert 46 <= hw.budget_gb <= 50
