"""Detect Apple Silicon hardware and resolve the active profile.

Reads hw.memsize via sysctl to size the model budget. The detected memory, not the
chip generation, predicts decode speed (docs/hardware.md).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from .profiles import model_budget_gb, select_profile

_GIB = 1024**3


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, check=True, timeout=5
        )
    except (subprocess.SubprocessError, OSError):
        return None
    val = out.stdout.strip()
    return val or None


@dataclass(frozen=True)
class Hardware:
    model: str | None
    cpu: str | None
    total_bytes: int
    total_gb: float
    budget_gb: float

    @property
    def profile(self) -> str:
        return select_profile(self.total_gb)


def detect() -> Hardware:
    raw = _sysctl("hw.memsize")
    total_bytes = int(raw) if raw and raw.isdigit() else 0
    total_gb = round(total_bytes / _GIB, 1)
    return Hardware(
        model=_sysctl("hw.model"),
        cpu=_sysctl("machdep.cpu.brand_string"),
        total_bytes=total_bytes,
        total_gb=total_gb,
        budget_gb=round(model_budget_gb(total_gb), 1),
    )


def resolve_profile(requested: str, total_gb: float) -> str:
    """Resolve the active profile name. 'auto' detects from memory; else pass through."""
    return select_profile(total_gb) if requested == "auto" else requested
