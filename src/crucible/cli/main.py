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
    host: str = typer.Option("", "--host", help="Override server.host."),
    port: int = typer.Option(0, "--port", help="Override server.port."),
) -> None:
    """Start the OpenAI-compatible gateway over the model manager (M2)."""
    import uvicorn

    from crucible.manager import ModelManager, make_loader, resolve_runtime
    from crucible.server import create_app

    try:
        reg = load_registry(config)
    except ConfigError as e:
        typer.secho(f"config error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e

    if not reg.models:
        typer.secho("registry declares no models to serve.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    hw = detect()
    active = resolve_profile(reg.profile, hw.total_gb)
    runtime = resolve_runtime(reg, active, hw.budget_gb)
    bind_host = host or reg.server.host
    bind_port = port or reg.server.port

    loader = make_loader(batching=reg.server.batching, max_kv_size=runtime.default_context)
    manager = ModelManager(reg, runtime, loader)
    typer.secho(
        f"profile {active}: ceiling {runtime.ceiling_gb} GB, "
        f"single_resident={runtime.single_resident}, batching={reg.server.batching}",
        fg=typer.colors.CYAN,
    )
    manager.warmup()  # eagerly load pinned models
    application = create_app(manager, runtime)
    typer.secho(
        f"serving {len(reg.models)} model(s) on http://{bind_host}:{bind_port} "
        f"[resident: {manager.resident_models() or 'none (lazy)'}]",
        fg=typer.colors.GREEN,
    )
    uvicorn.run(application, host=bind_host, port=bind_port, log_level="info")


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
