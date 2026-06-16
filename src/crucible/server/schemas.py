"""Request and response shapes for the OpenAI-compatible surface (docs/api.md).

Compatibility is the goal: stock OpenAI clients must work by changing only base_url.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from crucible.backends import SamplingParams
from crucible.config import Sampling

# --- chat ---


def _params(req, defaults: Sampling) -> SamplingParams:
    """Build SamplingParams, filling any field the request omitted from server defaults."""
    pick = lambda v, d: d if v is None else v  # noqa: E731
    return SamplingParams(
        max_tokens=req.max_tokens or defaults.max_tokens,
        temperature=pick(req.temperature, defaults.temperature),
        top_p=pick(req.top_p, defaults.top_p),
        repetition_penalty=pick(req.repetition_penalty, defaults.repetition_penalty),
        repetition_context_size=defaults.repetition_context_size,
        loop_guard=pick(getattr(req, "loop_guard", None), defaults.loop_guard),
        enable_thinking=pick(getattr(req, "enable_thinking", None), defaults.enable_thinking),
        stop=_as_list(req.stop),
    )


class ChatMessage(BaseModel):
    role: str
    # Text string, or OpenAI content-parts (vision parts are ignored until M6).
    content: str | list[dict] | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    repetition_penalty: float | None = Field(default=None, ge=1)
    loop_guard: bool | None = None
    enable_thinking: bool | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    stop: str | list[str] | None = None

    def sampling(self, defaults: Sampling) -> SamplingParams:
        return _params(self, defaults)

    def rendered_messages(self) -> list[dict]:
        return [{"role": m.role, "content": _flatten(m.content)} for m in self.messages]


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    repetition_penalty: float | None = Field(default=None, ge=1)
    loop_guard: bool | None = None
    enable_thinking: bool | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    stop: str | list[str] | None = None

    def sampling(self, defaults: Sampling) -> SamplingParams:
        return _params(self, defaults)

    def first_prompt(self) -> str:
        return self.prompt[0] if isinstance(self.prompt, list) else self.prompt


# --- helpers ---


class EmbeddingsRequest(BaseModel):
    model: str
    input: str | list[str]

    def texts(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else list(self.input)


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: int | None = Field(default=None, gt=0)


class RagIngestRequest(BaseModel):
    paths: str | list[str]


class RagQueryRequest(BaseModel):
    query: str
    rerank: bool | None = None
    top_k: int | None = Field(default=None, gt=0)
    top_n: int | None = Field(default=None, gt=0)


def _as_list(stop: str | list[str] | None) -> list[str]:
    if stop is None:
        return []
    return [stop] if isinstance(stop, str) else list(stop)


def _flatten(content: str | list[dict] | None) -> str:
    """Reduce content to text. Non-text parts (images) are dropped until M6."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            out.append(part.get("text", ""))
    return "".join(out)
