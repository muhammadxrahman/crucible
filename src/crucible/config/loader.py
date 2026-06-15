"""Load and validate config/models.yaml into the typed Registry."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from .schema import Registry


class ConfigError(Exception):
    """Raised when the registry file is missing, unparseable, or invalid."""


def load_registry(path: str | Path) -> Registry:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"registry file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"registry is not valid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"registry root must be a mapping, got {type(raw).__name__}")
    try:
        return Registry.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"registry failed validation:\n{e}") from e
