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


def test_sampling_defaults_and_override() -> None:
    # Defaults are chat-sane with a repetition penalty so any model terminates.
    default = Registry.model_validate({}).server.sampling
    assert default.temperature == 0.7
    assert default.repetition_penalty == 1.1
    assert default.top_p == 0.95
    # Config can override them per deployment.
    over = Registry.model_validate(
        {"server": {"sampling": {"temperature": 0.3, "repetition_penalty": 1.25}}}
    ).server.sampling
    assert over.temperature == 0.3
    assert over.repetition_penalty == 1.25


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


def test_append_model_persists_and_preserves_comments(tmp_path: Path) -> None:
    from crucible.config import ModelEntry
    from crucible.config.store import append_model

    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "# my registry — keep this comment\n"
        "server:\n"
        "  port: 8000  # inline note\n"
        "models:\n"
        "  - {path: mlx-community/Existing-4bit, type: lm, served_name: primary}\n"
    )
    append_model(
        cfg, ModelEntry(path="mlx-community/Llama-3.2-3B-4bit", type="lm", served_name="llama3")
    )

    text = cfg.read_text()
    assert "# my registry — keep this comment" in text  # comments survive the round-trip
    assert "# inline note" in text

    reg = load_registry(cfg)  # the new entry loads back as a valid registry
    by_name = {m.served_name: m for m in reg.models}
    assert by_name["primary"].path == "mlx-community/Existing-4bit"
    assert by_name["llama3"].path == "mlx-community/Llama-3.2-3B-4bit"


def test_append_model_creates_models_list_when_absent(tmp_path: Path) -> None:
    from crucible.config import ModelEntry
    from crucible.config.store import append_model

    cfg = tmp_path / "models.yaml"
    cfg.write_text("server:\n  port: 8000\n")
    append_model(cfg, ModelEntry(path="m/x", type="embedding", served_name="e", pin=True))

    reg = load_registry(cfg)
    assert reg.models[0].served_name == "e"
    assert reg.models[0].pin is True
