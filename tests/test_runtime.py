"""M2: runtime profile resolution combines config overrides with tier defaults."""

from crucible.config import Registry
from crucible.manager import resolve_runtime

_GIB = 1024**3


def reg(data: dict) -> Registry:
    return Registry.model_validate(data)


def test_config_profile_overrides_defaults() -> None:
    r = reg(
        {
            "profiles": {
                "pro64": {
                    "single_resident": False,
                    "default_context": 32768,
                    "kv_bits": 8,
                    "vision": True,
                }
            }
        }
    )
    rt = resolve_runtime(r, "pro64", budget_gb=48)
    assert rt.single_resident is False
    assert rt.vision is True
    assert rt.ceiling_bytes == int(48 * _GIB)  # no explicit ceiling -> hardware budget


def test_explicit_ceiling_wins_over_budget() -> None:
    r = reg({"server": {"memory_ceiling_gb": 30}})
    rt = resolve_runtime(r, "pro64", budget_gb=48)
    assert rt.ceiling_gb == 30.0


def test_tier_defaults_when_profile_absent_from_config() -> None:
    rt_small = resolve_runtime(reg({}), "air16", budget_gb=8)
    assert rt_small.single_resident is True
    assert rt_small.vision is False
    assert rt_small.kv_bits == 4

    rt_big = resolve_runtime(reg({}), "pro64", budget_gb=48)
    assert rt_big.single_resident is False
    assert rt_big.vision is True
    assert rt_big.kv_bits == 8
