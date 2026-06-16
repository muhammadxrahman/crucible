"""Persist runtime model additions back into the registry YAML.

When a model is added through the UI it is appended to the active config file so it survives
a restart. ruamel.yaml round-trips the document, preserving the user's comments and layout
(plain PyYAML would strip them).
"""

from __future__ import annotations

from pathlib import Path

from ruamel.yaml import YAML

from .schema import ModelEntry


def append_model(config_path: str | Path, entry: ModelEntry) -> None:
    """Append `entry` to the `models:` list of the registry file, keeping comments intact.
    Only fields the user chose are written; schema defaults (quant, etc.) are left implicit."""
    p = Path(config_path)
    yaml = YAML()
    yaml.preserve_quotes = True
    data = yaml.load(p.read_text()) if p.is_file() else None
    if data is None:
        data = {}
    models = data.get("models")
    if models is None:
        models = []
        data["models"] = models

    item = {"path": entry.path, "type": entry.type, "served_name": entry.served_name}
    if entry.pin:
        item["pin"] = True
    if entry.ttl_seconds is not None:
        item["ttl_seconds"] = entry.ttl_seconds
    models.append(item)

    with p.open("w") as f:
        yaml.dump(data, f)
