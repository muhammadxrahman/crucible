"""M7: packaging — launchd plist, the HTTP client, and the CLI (no GPU)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from crucible import service
from crucible.backends import Delta, Final, SamplingParams
from crucible.cli.main import app
from crucible.client import ClientError, CrucibleClient
from crucible.config import Registry
from crucible.manager import ModelManager, RuntimeProfile
from crucible.server import create_app

runner = CliRunner()


# --- launchd plist (pure) ---


def test_render_plist_runs_serve_at_login() -> None:
    xml = service.render_plist(
        uv_path="/opt/homebrew/bin/uv",
        working_dir="/home/me/crucible",
        config="config/models.yaml",
        log_dir="/home/me/crucible/.crucible/logs",
        label="com.crucible.mlxd",
    )
    assert "<string>com.crucible.mlxd</string>" in xml
    assert "<string>/opt/homebrew/bin/uv</string>" in xml
    assert "<string>serve</string>" in xml
    assert "<string>config/models.yaml</string>" in xml
    assert "<key>RunAtLoad</key>" in xml and "<true/>" in xml
    assert "<key>KeepAlive</key>" in xml
    assert "mlxd.err.log" in xml


def test_render_plist_escapes_xml() -> None:
    xml = service.render_plist(uv_path="/bin/uv", working_dir="/a&b", config="c.yaml", log_dir="/l")
    assert "/a&amp;b" in xml  # & escaped, no raw ampersand
    assert "/a&b<" not in xml


# --- client + CLI against an in-process app ---


class FakeEngine:
    served_name = "primary"

    def stream(self, messages: list[dict], params: SamplingParams) -> Iterator:
        yield Delta("hi")
        yield Final(prompt_tokens=1, completion_tokens=1, finish_reason="stop")


@pytest.fixture
def http_client() -> Iterator[CrucibleClient]:
    reg = Registry.model_validate(
        {"models": [{"path": "f/p", "type": "lm", "served_name": "primary", "pin": True}]}
    )
    runtime = RuntimeProfile(
        name="pro64",
        ceiling_bytes=10**12,
        single_resident=False,
        default_context=8192,
        kv_bits=8,
        vision=True,
    )
    manager = ModelManager(reg, runtime, lambda e: (FakeEngine(), 2_000_000))
    tc = TestClient(create_app(manager, runtime))
    yield CrucibleClient(client=tc)


def test_client_list_load_unload_pin(http_client: CrucibleClient) -> None:
    assert http_client.health()["status"] == "ok"

    models = http_client.list_models()
    assert models[0]["id"] == "primary"

    assert http_client.load("primary")["state"] == "resident"
    assert http_client.unload("primary")["state"] == "available"
    assert http_client.pin("primary", True)["pinned"] is True


def test_client_unknown_model_raises(http_client: CrucibleClient) -> None:
    with pytest.raises(ClientError, match="not found"):
        http_client.load("ghost")


def test_cli_validate_ok(tmp_path) -> None:
    cfg = tmp_path / "models.yaml"
    cfg.write_text("models:\n  - {path: m/x, type: lm, served_name: primary}\n")
    result = runner.invoke(app, ["validate", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "valid" in result.stdout
    assert "1 lm" in result.stdout


def test_cli_validate_rejects_bad_config(tmp_path) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("models:\n  - {path: m/x, type: lm, served_name: a, bogus: 1}\n")
    result = runner.invoke(app, ["validate", "--config", str(cfg)])
    assert result.exit_code == 2
    assert "invalid config" in result.stderr


def _pull_config(tmp_path) -> str:
    cfg = tmp_path / "m.yaml"
    cfg.write_text(
        "models:\n"
        "  - {path: org/a, type: lm, served_name: a}\n"
        "  - {path: org/b, type: embedding, served_name: b}\n"
    )
    return str(cfg)


def test_cli_pull_downloads_each_model(monkeypatch, tmp_path) -> None:
    pulled: list[str] = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo, **k: pulled.append(repo))
    result = runner.invoke(app, ["pull", "--config", _pull_config(tmp_path)])
    assert result.exit_code == 0
    assert pulled == ["org/a", "org/b"]


def test_cli_pull_specific_model(monkeypatch, tmp_path) -> None:
    pulled: list[str] = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo, **k: pulled.append(repo))
    result = runner.invoke(app, ["pull", "b", "--config", _pull_config(tmp_path)])
    assert result.exit_code == 0
    assert pulled == ["org/b"]


def test_cli_pull_unknown_name_errors(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo, **k: None)
    result = runner.invoke(app, ["pull", "ghost", "--config", _pull_config(tmp_path)])
    assert result.exit_code == 2


def test_cli_train_is_reserved_for_m9() -> None:
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 1
    assert "deferred to M9" in result.stderr


def test_cli_models_list_via_client(monkeypatch, http_client: CrucibleClient) -> None:
    # Point the CLI's client factory at the in-process client.
    monkeypatch.setattr("crucible.cli.main._client", lambda server: http_client)
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "primary" in result.stdout
    assert "lm" in result.stdout
