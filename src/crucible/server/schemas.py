"""Request and response shapes for the OpenAI-compatible surface (docs/api.md).

Compatibility is the goal: stock OpenAI clients must work by changing only base_url.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from crucible.backends import SamplingParams

# --- chat ---


class ChatMessage(BaseModel):
    role: str
    # Text string, or OpenAI content-parts (vision parts are ignored until M6).
    content: str | list[dict] | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int | None = Field(default=None, gt=0)
    stop: str | list[str] | None = None

    def sampling(self, default_max_tokens: int) -> SamplingParams:
        return SamplingParams(
            max_tokens=self.max_tokens or default_max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=_as_list(self.stop),
        )

    def rendered_messages(self) -> list[dict]:
        return [{"role": m.role, "content": _flatten(m.content)} for m in self.messages]


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    stream: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int | None = Field(default=None, gt=0)
    stop: str | list[str] | None = None

    def sampling(self, default_max_tokens: int) -> SamplingParams:
        return SamplingParams(
            max_tokens=self.max_tokens or default_max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=_as_list(self.stop),
        )

    def first_prompt(self) -> str:
        return self.prompt[0] if isinstance(self.prompt, list) else self.prompt


# --- helpers ---


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
