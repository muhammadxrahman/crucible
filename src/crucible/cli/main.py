"""The mlxd command-line entry point.

M0 implements `profile`. `serve`, `models`, and `bench` are declared but report that
their milestone is not yet built, so the CLI surface is stable from the start.
"""

from __future__ import annotations

from pathlib import Path

import typer

from crucible.config import ConfigError, load_registry
from crucible.hardware import detect, resolve_profile

app = typer.Typer(add_completion=False, help="Crucible MLX inference platform.")

DEFAULT_CONFIG = Path("config/models.yaml")


def _requested_profile(config: Path) -> str:
    """Read the requested profile from config if present; default to auto."""
    if not config.is_file():
        return "auto"
    try:
        return load_registry(config).profile
    except ConfigError as e:
        typer.secho(f"config error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e


@app.command()
def profile(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to models.yaml."),
    requested: str = typer.Option(
        "", "--profile", "-p", help="Override profile (else config, else auto)."
    ),
) -> None:
    """Show detected hardware and the active profile."""
    hw = detect()
    want = requested or _requested_profile(config)
    active = resolve_profile(want, hw.total_gb)

    typer.secho("Hardware", bold=True)
    typer.echo(f"  model:        {hw.model or 'unknown'}")
    typer.echo(f"  cpu:          {hw.cpu or 'unknown'}")
    typer.echo(f"  memory:       {hw.total_gb:g} GB")
    typer.echo(f"  model budget: {hw.budget_gb:g} GB")
    typer.secho("Profile", bold=True)
    typer.echo(f"  requested:    {want}")
    typer.echo(f"  active:       {active}")


def _not_yet(name: str, milestone: str) -> None:
    typer.secho(
        f"`mlxd {name}` is not built yet (arrives in {milestone}).",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(code=1)


@app.command()
def serve(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to models.yaml."),
    model: str = typer.Option("", "--model", "-m", help="served_name to load (default: first lm)."),
    host: str = typer.Option("", "--host", help="Override server.host."),
    port: int = typer.Option(0, "--port", help="Override server.port."),
) -> None:
    """Start the OpenAI-compatible gateway on one text model (M1)."""
    import uvicorn

    from crucible.backends.text import MLXTextEngine
    from crucible.server import create_app

    try:
        reg = load_registry(config)
    except ConfigError as e:
        typer.secho(f"config error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e

    entry = _pick_text_model(reg, model)
    if entry is None:
        typer.secho(
            "no text (type: lm) model to serve; add one to the registry or pass --model.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    hw = detect()
    active = resolve_profile(reg.profile, hw.total_gb)
    bind_host = host or reg.server.host
    bind_port = port or reg.server.port

    typer.secho(f"loading '{entry.served_name}' ({entry.path}) ...", fg=typer.colors.CYAN)
    engine = MLXTextEngine(entry.path, entry.served_name)
    application = create_app(engine, active)
    typer.secho(
        f"serving '{entry.served_name}' on http://{bind_host}:{bind_port} [profile: {active}]",
        fg=typer.colors.GREEN,
    )
    uvicorn.run(application, host=bind_host, port=bind_port, log_level="info")


def _pick_text_model(reg, name: str):  # noqa: ANN001
    """Resolve the text model: explicit served_name, else first pinned lm, else first lm."""
    lms = [m for m in reg.models if m.type == "lm"]
    if name:
        return next((m for m in lms if m.served_name == name), None)
    return next((m for m in lms if m.pin), None) or (lms[0] if lms else None)


@app.command()
def models() -> None:
    """List, load, unload, and pin models (M2+)."""
    _not_yet("models", "M2")


@app.command()
def bench() -> None:
    """Run the benchmark harness (M4+)."""
    _not_yet("bench", "M4")


if __name__ == "__main__":
    app()
