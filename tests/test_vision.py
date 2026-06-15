"""M6: vision message parsing, image cache, and image-request routing (no GPU)."""

from __future__ import annotations

import base64
from collections import OrderedDict
from pathlib import Path

from fastapi.testclient import TestClient

from crucible.backends import Delta, Final
from crucible.backends.images import (
    ImageRef,
    extract_images,
    flatten_text,
    materialize,
    parse_image_url,
    text_messages,
)
from crucible.backends.vision import MLXVLMEngine
from crucible.config import Registry
from crucible.manager import ModelManager, RuntimeProfile
from crucible.server import create_app

# A 1x1 PNG.
_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="  # noqa: E501
_DATA_URL = f"data:image/png;base64,{_PNG_B64}"


def _img_message(text: str, url: str = _DATA_URL) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": url}},
        ],
    }


# --- parsing ---


def test_parse_data_url() -> None:
    ref = parse_image_url(_DATA_URL)
    assert ref.kind == "data"
    assert ref.payload == _PNG_B64
    assert parse_image_url(_DATA_URL).sha == ref.sha  # stable hash


def test_parse_http_url() -> None:
    ref = parse_image_url("https://example.com/cat.png")
    assert ref.kind == "url"
    assert ref.payload == "https://example.com/cat.png"


def test_extract_images_in_order() -> None:
    msgs = [
        {"role": "system", "content": "be brief"},
        _img_message("what is this?", "https://a/1.png"),
        {"role": "assistant", "content": "a thing"},
        _img_message("and this?", "https://a/2.png"),
    ]
    refs = extract_images(msgs)
    assert [r.payload for r in refs] == ["https://a/1.png", "https://a/2.png"]


def test_extract_images_none_for_text() -> None:
    assert extract_images([{"role": "user", "content": "hello"}]) == []


def test_flatten_and_text_messages() -> None:
    assert flatten_text("plain") == "plain"
    assert flatten_text([{"type": "text", "text": "a"}, {"type": "image_url"}]) == "a"
    assert flatten_text(None) == ""
    tm = text_messages([_img_message("describe")])
    assert tm == [{"role": "user", "content": "describe"}]


def test_materialize_data_writes_file_and_url_passthrough(tmp_path: Path) -> None:
    ref = parse_image_url(_DATA_URL)
    path = materialize(ref)
    assert Path(path).is_file()
    assert Path(path).read_bytes() == base64.b64decode(_PNG_B64)
    assert materialize(ImageRef("url", "http://x/y.png", "abc")) == "http://x/y.png"


# --- VLM engine image cache (bypassing model load) ---


def _bare_vlm(cache_size: int = 2) -> MLXVLMEngine:
    e = object.__new__(MLXVLMEngine)
    e._img_cache = OrderedDict()
    e._hits = 0
    e._misses = 0
    e._cached_tokens = 0
    e._cache_size = cache_size
    return e


def test_image_cache_hit_and_miss() -> None:
    e = _bare_vlm()
    a = ImageRef("url", "http://x/a.png", "a")
    e._image_path(a)
    e._image_path(a)  # same image -> hit
    assert e.stats() == {
        "vision_cache_hits": 1,
        "vision_cache_misses": 1,
        "vision_cached_tokens": 0,
    }


def test_image_cache_lru_eviction() -> None:
    e = _bare_vlm(cache_size=2)
    for name in ["a", "b", "c"]:
        e._image_path(ImageRef("url", f"http://x/{name}.png", name))
    assert "a" not in e._img_cache and {"b", "c"} <= set(e._img_cache)
    assert e._misses == 3


# --- routing through the gateway ---


class FakeVLM:
    type = "vlm"

    def __init__(self) -> None:
        self.seen: dict[str, bool] = {}
        self.hits = 0
        self.calls = 0

    def stream_vision(self, messages, params, images):
        self.calls += 1
        for ref in images:
            if ref.sha in self.seen:
                self.hits += 1
            else:
                self.seen[ref.sha] = True
        yield Delta("A small image.")
        yield Final(prompt_tokens=20, completion_tokens=3, finish_reason="stop")

    def stats(self) -> dict:
        return {"vision_cache_hits": self.hits}


class FakeGen:
    type = "lm"

    def stream(self, messages, params):
        yield Delta("text answer")
        yield Final(prompt_tokens=5, completion_tokens=2, finish_reason="stop")


def _client(with_vlm: bool = True):
    models = [{"path": "f/g", "type": "lm", "served_name": "gen"}]
    if with_vlm:
        models.append({"path": "f/v", "type": "vlm", "served_name": "vision"})
    reg = Registry.model_validate({"models": models})
    runtime = RuntimeProfile(
        name="pro64",
        ceiling_bytes=10**12,
        single_resident=False,
        default_context=8192,
        kv_bits=8,
        vision=True,
    )
    engines = {"gen": FakeGen(), "vision": FakeVLM()}

    def loader(entry):
        return engines[entry.served_name], 10

    manager = ModelManager(reg, runtime, loader)
    return TestClient(create_app(manager, runtime)), engines


def test_image_request_routes_to_vlm() -> None:
    c, engines = _client()
    body = c.post(
        "/v1/chat/completions",
        json={"model": "vision", "messages": [_img_message("what is this?")]},
    ).json()
    assert body["choices"][0]["message"]["content"] == "A small image."
    assert engines["vision"].calls == 1


def test_image_request_auto_routes_from_text_model() -> None:
    # client names a text model but sends an image -> routed to the VLM
    c, engines = _client()
    c.post("/v1/chat/completions", json={"model": "gen", "messages": [_img_message("hi")]})
    assert engines["vision"].calls == 1


def test_text_request_uses_text_engine() -> None:
    c, _ = _client()
    body = c.post(
        "/v1/chat/completions",
        json={"model": "gen", "messages": [{"role": "user", "content": "hello"}]},
    ).json()
    assert body["choices"][0]["message"]["content"] == "text answer"


def test_image_request_without_vlm_returns_400() -> None:
    c, _ = _client(with_vlm=False)
    r = c.post(
        "/v1/chat/completions",
        json={"model": "gen", "messages": [_img_message("hi")]},
    )
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "no_vision_model"


def test_multi_turn_same_image_reuses_cache_in_metrics() -> None:
    c, engines = _client()
    for _ in range(2):
        c.post(
            "/v1/chat/completions",
            json={"model": "vision", "messages": [_img_message("describe", "https://a/same.png")]},
        )
    assert engines["vision"].hits == 1
    summary = c.get("/metrics/summary").json()
    assert summary["current"]["vision_cache_hits"] == 1
