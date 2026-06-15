"""M0 acceptance: config validates a good registry and rejects malformed ones."""

from pathlib import Path

import pytest

from crucible.config import ConfigError, Registry, load_registry


def test_real_registry_validates(real_config: Path) -> None:
    reg = load_registry(real_config)
    assert isinstance(reg, Registry)
    assert {m.served_name for m in reg.models} == {"primary", "vision", "embed", "rerank"}
    assert reg.profiles["pro64"].vision is True
    assert reg.profiles["air16"].vision is False


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_registry(tmp_path / "nope.yaml")


def test_unknown_key_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "models:\n"
        "  - path: m/x\n"
        "    type: lm\n"
        "    served_name: a\n"
        "    bogus_key: 1\n"  # extra=forbid must reject this
    )
    with pytest.raises(ConfigError, match="validation"):
        load_registry(bad)


def test_duplicate_served_name_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "dupe.yaml"
    bad.write_text(
        "models:\n"
        "  - {path: m/x, type: lm, served_name: dup}\n"
        "  - {path: m/y, type: lm, served_name: dup}\n"
    )
    with pytest.raises(ConfigError, match="duplicate served_name"):
        load_registry(bad)


def test_bad_model_type_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "type.yaml"
    bad.write_text("models:\n  - {path: m/x, type: diffusion, served_name: a}\n")
    with pytest.raises(ConfigError, match="validation"):
        load_registry(bad)


def test_undefined_active_profile_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "prof.yaml"
    bad.write_text(
        "profile: pro999\n"
        "profiles:\n"
        "  pro64: {single_resident: false, default_context: 32768, kv_bits: 8, vision: true}\n"
    )
    with pytest.raises(ConfigError, match="not defined in profiles"):
        load_registry(bad)
