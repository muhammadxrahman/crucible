"""The mlxd command-line entry point.

Commands: `serve`, `pull` (pre-download model weights), `models` (list/load/unload/pin via
the running server), `bench`, `profile`, `validate`, `service` (launchd autostart), and
`train` (reserved for M9).
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


@app.command()
def serve(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to models.yaml."),
    host: str = typer.Option("", "--host", help="Override server.host."),
    port: int = typer.Option(0, "--port", help="Override server.port."),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the web UI in a browser once the server is ready."
    ),
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
    pinned = [m.served_name for m in reg.models if m.pin]
    if pinned:
        typer.secho(
            f"warming up pinned model(s) {pinned} (downloading if not cached; "
            "pre-fetch with `mlxd pull` to avoid the wait) ...",
            fg=typer.colors.CYAN,
        )
    for name, err in manager.warmup():  # resilient: a failed model is skipped, not fatal
        typer.secho(
            f"  warning: '{name}' did not load ({err}); will retry lazily.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    from crucible.rag import RagPipeline, resolve_rag_roles

    roles = resolve_rag_roles(reg)
    rag = None
    if roles["embed_name"] and roles["generator_name"]:
        rag = RagPipeline(
            manager,
            reg.rag,
            embed_name=roles["embed_name"],
            generator_name=roles["generator_name"],
            rerank_name=roles["rerank_name"],
        )
        typer.secho(
            f"RAG enabled: embed={roles['embed_name']} generator={roles['generator_name']} "
            f"rerank={roles['rerank_name'] or 'off'}",
            fg=typer.colors.CYAN,
        )
    import os

    from crucible.history import HistoryStore

    history_db = os.environ.get("CRUCIBLE_HISTORY_DB", str(Path(".crucible") / "history.db"))
    history = HistoryStore(history_db)
    application = create_app(
        manager,
        runtime,
        rag,
        sampling=reg.server.sampling,
        config_path=config,
        history=history,
    )
    ui_host = "127.0.0.1" if bind_host in ("0.0.0.0", "") else bind_host
    url = f"http://{ui_host}:{bind_port}/"
    typer.secho(
        f"serving {len(reg.models)} model(s) on {url} "
        f"[resident: {manager.resident_models() or 'none (lazy)'}]",
        fg=typer.colors.GREEN,
    )
    if open_browser:
        _open_browser_when_ready(url)
    uvicorn.run(application, host=bind_host, port=bind_port, log_level="info")


def _open_browser_when_ready(url: str) -> None:
    """Open the web UI once the server is actually listening (polls /healthz in a thread)."""
    import threading

    def wait_and_open() -> None:
        import time
        import webbrowser

        import httpx

        health = url.rstrip("/") + "/healthz"
        for _ in range(150):  # up to ~30s
            try:
                if httpx.get(health, timeout=1).status_code == 200:
                    webbrowser.open(url)
                    return
            except httpx.HTTPError:
                time.sleep(0.2)

    threading.Thread(target=wait_and_open, daemon=True).start()


@app.command()
def bench(
    spec: Path = typer.Argument(..., help="Path to a benchmark spec YAML."),
) -> None:
    """Run the benchmark harness over a spec and write a Markdown report."""
    from crucible.benchmark import run_spec

    if not spec.is_file():
        typer.secho(f"spec not found: {spec}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    typer.secho(f"running benchmark from {spec} ...", fg=typer.colors.CYAN)
    report = run_spec(spec)
    typer.secho(f"report written: {report}", fg=typer.colors.GREEN)


@app.command()
def validate(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to models.yaml."),
) -> None:
    """Validate the registry and report the active profile, without starting the server."""
    try:
        reg = load_registry(config)
    except ConfigError as e:
        typer.secho(f"invalid config: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    hw = detect()
    active = resolve_profile(reg.profile, hw.total_gb)
    by_type: dict[str, int] = {}
    for m in reg.models:
        by_type[m.type] = by_type.get(m.type, 0) + 1
    summary = ", ".join(f"{n} {t}" for t, n in sorted(by_type.items())) or "no models"
    typer.secho(f"OK: {config} valid ({summary}); active profile {active}.", fg=typer.colors.GREEN)


@app.command()
def pull(
    names: list[str] = typer.Argument(None, help="served_name(s) to pull; default: all in config."),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to models.yaml."),
) -> None:
    """Download model weights into the local cache, with progress, before serving.

    Run this first for large models (e.g. the 30B): serving then loads from cache instead of
    blocking on a multi-gigabyte download at startup.
    """
    try:
        reg = load_registry(config)
    except ConfigError as e:
        typer.secho(f"config error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e

    entries = reg.models
    if names:
        entries = [m for m in reg.models if m.served_name in names]
        missing = set(names) - {m.served_name for m in entries}
        if missing:
            typer.secho(f"unknown served_name(s): {sorted(missing)}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2)
    if not entries:
        typer.secho("nothing to pull.", fg=typer.colors.YELLOW)
        return

    from huggingface_hub import snapshot_download

    for e in entries:
        typer.secho(f"pulling {e.served_name} ({e.path}) ...", fg=typer.colors.CYAN)
        try:
            snapshot_download(e.path)
        except Exception as err:  # noqa: BLE001
            typer.secho(f"  failed: {err}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from err
    typer.secho(f"pulled {len(entries)} model(s) into the local cache.", fg=typer.colors.GREEN)


@app.command()
def train() -> None:
    """Fine-tune a model with LoRA (deferred to M9)."""
    typer.secho(
        "`mlxd train` is deferred to M9 (LoRA fine-tuning). The registry `adapters` field "
        "and this command are reserved for it.",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(code=1)


# --- models: an HTTP client to a running server ---

models_app = typer.Typer(add_completion=False, help="List, load, unload, and pin models.")
app.add_typer(models_app, name="models")

_SERVER_OPT = typer.Option("http://127.0.0.1:8000", "--server", "-s", help="Server base URL.")


def _client(server: str):
    from crucible.client import CrucibleClient

    return CrucibleClient(server)


def _run_client(server: str, fn) -> None:
    from crucible.client import ClientError

    try:
        with _client(server) as c:
            fn(c)
    except ClientError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e


@models_app.command("list")
def models_list(server: str = _SERVER_OPT) -> None:
    """List models and their residency."""

    def go(c) -> None:
        for m in c.list_models():
            mark = "*" if m.get("pinned") else " "
            mb = m.get("resident_mb", 0)
            typer.echo(
                f"{mark} {m['id']:<16} {m['type']:<10} {m.get('state', '?'):<10} {mb:>8.1f} MB"
            )

    _run_client(server, go)


@models_app.command("load")
def models_load(name: str, server: str = _SERVER_OPT) -> None:
    """Load a model into memory."""
    _run_client(server, lambda c: typer.echo(f"{name}: {c.load(name)['state']}"))


@models_app.command("unload")
def models_unload(name: str, server: str = _SERVER_OPT) -> None:
    """Evict a model from memory."""
    _run_client(server, lambda c: typer.echo(f"{name}: {c.unload(name)['state']}"))


@models_app.command("pin")
def models_pin(
    name: str,
    off: bool = typer.Option(False, "--off", help="Unpin instead of pin."),
    server: str = _SERVER_OPT,
) -> None:
    """Pin (or unpin) a model so it is never evicted."""
    _run_client(server, lambda c: typer.echo(f"{name}: pinned={c.pin(name, not off)['pinned']}"))


# --- service: native autostart via launchd ---

service_app = typer.Typer(add_completion=False, help="Run the engine on login via launchd.")
app.add_typer(service_app, name="service")


@service_app.command("install")
def service_install(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to models.yaml."),
) -> None:
    """Install and load a LaunchAgent so the server starts on login (native, not Docker)."""
    import shutil

    from crucible import service

    uv_path = shutil.which("uv")
    if not uv_path:
        typer.secho("uv not found on PATH; install uv first.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    cwd = Path.cwd()
    log_dir = cwd / ".crucible" / "logs"
    path = service.install(
        uv_path=uv_path,
        working_dir=str(cwd),
        config=str(config),
        log_dir=str(log_dir),
    )
    typer.secho(f"installed LaunchAgent at {path}", fg=typer.colors.GREEN)
    typer.echo(f"logs: {log_dir}")


@service_app.command("uninstall")
def service_uninstall() -> None:
    """Unload and remove the LaunchAgent."""
    from crucible import service

    removed = service.uninstall()
    typer.secho("removed" if removed else "no LaunchAgent installed", fg=typer.colors.GREEN)


@service_app.command("status")
def service_status() -> None:
    """Show whether the LaunchAgent is loaded."""
    from crucible import service

    typer.echo(service.status())


if __name__ == "__main__":
    app()
