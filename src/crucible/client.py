"""A thin HTTP client for the Crucible server.

The CLI's `models` commands are API clients: a separate `mlxd` invocation cannot reach the
running server's in-memory manager, so it talks to the same public endpoints any client
uses. Accepts an injectable transport so it can be driven against an in-process ASGI app
in tests.
"""

from __future__ import annotations

import httpx

DEFAULT_BASE = "http://127.0.0.1:8000"


class ClientError(Exception):
    pass


class CrucibleClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE,
        *,
        transport=None,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ):
        # `client` allows injecting an in-process client (e.g. FastAPI TestClient) in tests.
        self._c = client or httpx.Client(base_url=base_url, transport=transport, timeout=timeout)

    def __enter__(self) -> CrucibleClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._c.close()

    def health(self) -> dict:
        return self._get("/healthz")

    def list_models(self) -> list[dict]:
        return self._get("/v1/models")["data"]

    def load(self, served_name: str) -> dict:
        return self._post("/admin/models/load", {"served_name": served_name})

    def unload(self, served_name: str) -> dict:
        return self._post("/admin/models/unload", {"served_name": served_name})

    def pin(self, served_name: str, pinned: bool = True) -> dict:
        return self._post("/admin/models/pin", {"served_name": served_name, "pinned": pinned})

    # --- internals ---

    def _get(self, path: str) -> dict:
        return self._handle(lambda: self._c.get(path))

    def _post(self, path: str, body: dict) -> dict:
        return self._handle(lambda: self._c.post(path, json=body))

    def _handle(self, call) -> dict:
        try:
            r = call()
        except httpx.HTTPError as e:
            raise ClientError(f"cannot reach server: {e}") from e
        if r.status_code >= 400:
            try:
                msg = r.json()["error"]["message"]
            except Exception:
                msg = r.text
            raise ClientError(msg)
        return r.json()
