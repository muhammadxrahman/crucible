"""List MLX models already in the local Hugging Face cache, for the UI's 'add model' picker.

This does not download anything; it only surfaces what is already on disk so a downloaded
model can be served without editing config by hand. Type is a best-effort guess from the
repo name, which the user can override.
"""

from __future__ import annotations


def guess_type(repo_id: str) -> str:
    """Heuristic model type from the repo name (the UI lets the user override it)."""
    s = repo_id.lower()
    if "rerank" in s:
        return "rerank"
    if "-vl" in s or "vl-" in s or "vision" in s:
        return "vlm"
    if any(k in s for k in ("embed", "bge", "gte", "e5", "minilm", "nomic")):
        return "embedding"
    return "lm"


def _has_weights(repo) -> bool:
    """A real model has weight files; incomplete/metadata-only pulls do not."""
    for rev in repo.revisions:
        for f in rev.files:
            if f.file_name.endswith((".safetensors", ".npz")):
                return True
    return False


def available_models(registered_paths: set[str] | None = None) -> list[dict]:
    """Cached `model` repos that actually have weights, sorted by name. Each item:
    {repo_id, size_bytes, size_str, guessed_type, registered}."""
    from huggingface_hub import scan_cache_dir

    registered = registered_paths or set()
    try:
        info = scan_cache_dir()
    except Exception:  # noqa: BLE001 - no/unreadable cache -> empty list, not a crash
        return []
    out = [
        {
            "repo_id": repo.repo_id,
            "size_bytes": repo.size_on_disk,
            "size_str": repo.size_on_disk_str,
            "guessed_type": guess_type(repo.repo_id),
            "registered": repo.repo_id in registered,
        }
        for repo in info.repos
        if repo.repo_type == "model" and _has_weights(repo)
    ]
    out.sort(key=lambda m: m["repo_id"].lower())
    return out
