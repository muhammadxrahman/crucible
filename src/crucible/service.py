"""launchd integration: run the engine native on login, not in Docker.

The engine must run on the host to reach Metal, so a LaunchAgent (not a container) is the
right autostart mechanism. `render_plist` is pure and tested; install/uninstall shell out
to `launchctl`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

LABEL = "com.crucible.mlxd"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def render_plist(
    *,
    uv_path: str,
    working_dir: str,
    config: str,
    log_dir: str,
    label: str = LABEL,
) -> str:
    """Render a LaunchAgent plist that runs `uv run mlxd serve` at login."""
    args = [uv_path, "run", "mlxd", "serve", "--config", config]
    arg_xml = "\n".join(f"      <string>{escape(a)}</string>" for a in args)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{escape(label)}</string>
    <key>ProgramArguments</key>
    <array>
{arg_xml}
    </array>
    <key>WorkingDirectory</key>
    <string>{escape(working_dir)}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{escape(log_dir)}/mlxd.out.log</string>
    <key>StandardErrorPath</key>
    <string>{escape(log_dir)}/mlxd.err.log</string>
</dict>
</plist>
"""


def install(*, uv_path: str, working_dir: str, config: str, log_dir: str) -> Path:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    path = plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_plist(uv_path=uv_path, working_dir=working_dir, config=config, log_dir=log_dir)
    )
    _launchctl("unload", str(path), check=False)  # reload if already present
    _launchctl("load", str(path))
    return path


def uninstall() -> bool:
    path = plist_path()
    if not path.exists():
        return False
    _launchctl("unload", str(path), check=False)
    path.unlink()
    return True


def status() -> str:
    out = _launchctl("list", check=False)
    for line in out.splitlines():
        if LABEL in line:
            return f"running: {line.strip()}"
    return "not loaded"


def _launchctl(*args: str, check: bool = True) -> str:
    result = subprocess.run(["launchctl", *args], capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"launchctl {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout
