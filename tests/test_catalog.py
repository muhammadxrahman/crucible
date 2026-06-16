"""The HF-cache catalog that backs the UI's 'add downloaded model' picker (no GPU/network)."""

from __future__ import annotations

from dataclasses import dataclass, field

from crucible.manager import catalog


def test_guess_type_from_repo_name() -> None:
    assert catalog.guess_type("mlx-community/Qwen3-Reranker-0.6B-4bit") == "rerank"
    assert catalog.guess_type("mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit") == "vlm"
    assert catalog.guess_type("mlx-community/Qwen2-VL-2B-Instruct-4bit") == "vlm"
    assert catalog.guess_type("mlx-community/bge-small-en-v1.5-bf16") == "embedding"
    assert catalog.guess_type("mlx-community/Qwen3-30B-A3B-4bit") == "lm"
    assert catalog.guess_type("mlx-community/Llama-3.2-3B-Instruct-4bit") == "lm"


@dataclass
class _File:
    file_name: str


@dataclass
class _Rev:
    files: list[_File]


@dataclass
class _Repo:
    repo_id: str
    repo_type: str
    size_on_disk: int
    size_on_disk_str: str
    revisions: list[_Rev] = field(default_factory=list)


@dataclass
class _Cache:
    repos: list[_Repo]


def _weighted(repo_id: str, size: int) -> _Repo:
    return _Repo(repo_id, "model", size, f"{size}B", [_Rev([_File("model.safetensors")])])


def test_available_models_filters_stubs_and_datasets(monkeypatch) -> None:
    real = _weighted("mlx-community/Llama-3.2-3B-Instruct-4bit", 1_800_000_000)
    stub = _Repo(
        "mlx-community/Broken-Stub-4bit", "model", 1200, "1.2K", [_Rev([_File("config.json")])]
    )
    dataset = _Repo("some/dataset", "dataset", 9_000, "9K", [_Rev([_File("data.safetensors")])])
    registered = _weighted("mlx-community/Qwen3-30B-A3B-4bit", 17_000_000_000)

    # available_models does `from huggingface_hub import scan_cache_dir` at call time.
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "scan_cache_dir", lambda: _Cache([real, stub, dataset, registered])
    )

    out = catalog.available_models({"mlx-community/Qwen3-30B-A3B-4bit"})
    ids = [m["repo_id"] for m in out]
    assert "mlx-community/Llama-3.2-3B-Instruct-4bit" in ids  # real model kept
    assert "mlx-community/Broken-Stub-4bit" not in ids  # no weights -> dropped
    assert "some/dataset" not in ids  # not a model repo
    by_id = {m["repo_id"]: m for m in out}
    assert by_id["mlx-community/Llama-3.2-3B-Instruct-4bit"]["guessed_type"] == "lm"
    assert by_id["mlx-community/Qwen3-30B-A3B-4bit"]["registered"] is True


def test_available_models_empty_when_cache_unreadable(monkeypatch) -> None:
    import huggingface_hub

    def boom():
        raise OSError("no cache")

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", boom)
    assert catalog.available_models() == []
