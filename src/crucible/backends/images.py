"""Parse OpenAI vision message content and materialize images for mlx-vlm.

Accepts the standard content-parts shape: `{"type": "image_url", "image_url": {"url": ...}}`,
where the url is an HTTP(S) URL or a base64 `data:` URL. Pure and side-effect-free except
`materialize`, which writes a data URL to a temp file (mlx-vlm loads images by path/URL).
"""

from __future__ import annotations

import base64
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageRef:
    kind: str  # "data" | "url"
    payload: str  # base64 body (data) or the URL (url)
    sha: str  # content hash, used as the vision-cache key


def parse_image_url(url: str) -> ImageRef:
    if url.startswith("data:"):
        body = url.split(",", 1)[1] if "," in url else ""
        return ImageRef("data", body, _sha(body))
    return ImageRef("url", url, _sha(url))


def extract_images(messages: list[dict]) -> list[ImageRef]:
    """Pull every image_url part out of a list of OpenAI messages, in order."""
    refs: list[ImageRef] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if url:
                    refs.append(parse_image_url(url))
    return refs


def flatten_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "".join(parts)


def text_messages(messages: list[dict]) -> list[dict]:
    """Drop image parts, keep text, so the chat template can place image tokens itself."""
    return [
        {"role": m.get("role", "user"), "content": flatten_text(m.get("content"))} for m in messages
    ]


def materialize(ref: ImageRef) -> str:
    """Return a path or URL that mlx-vlm can load. Data URLs are written to a temp file."""
    if ref.kind == "url":
        return ref.payload
    path = Path(tempfile.gettempdir()) / f"crucible-img-{ref.sha}.bin"
    if not path.exists():
        path.write_bytes(base64.b64decode(ref.payload))
    return str(path)


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:16]
