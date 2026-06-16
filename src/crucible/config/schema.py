"""Pydantic schema for the model registry and hardware profiles (config/models.yaml).

Source code never hardcodes a model path or a memory limit; both come from here.
Unknown keys are rejected so a malformed registry fails loudly at boot.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ModelType = Literal["lm", "vlm", "embedding", "rerank"]
Quant = Literal[4, 8]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelEntry(_Strict):
    path: str
    type: ModelType
    served_name: str
    quant: Quant = 4
    context_length: int | None = Field(default=None, gt=0)
    pin: bool = False
    ttl_seconds: int | None = Field(default=None, gt=0)
    adapters: list[str] = Field(default_factory=list)


class Sampling(_Strict):
    """Default generation settings, applied when a request omits them. Chat-sane and with a
    repetition penalty so any model terminates instead of looping; overridable per request."""

    temperature: float = Field(default=0.7, ge=0)
    top_p: float = Field(default=0.95, gt=0, le=1)
    repetition_penalty: float = Field(default=1.1, ge=1)
    repetition_context_size: int = Field(default=20, gt=0)
    loop_guard: bool = True  # hard-stop runaway repetition loops
    enable_thinking: bool = False  # off -> reasoning models answer directly (no <think>)
    max_tokens: int = Field(default=512, gt=0)


class ServerConfig(_Strict):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, gt=0, lt=65536)
    memory_ceiling_gb: float | None = Field(default=None, gt=0)
    batching: bool = True
    sampling: Sampling = Field(default_factory=lambda: Sampling())


class ProfileSpec(_Strict):
    single_resident: bool
    default_context: int = Field(gt=0)
    kv_bits: Quant
    vision: bool
    roles: dict[str, str] = Field(default_factory=dict)


class RagConfig(_Strict):
    embed_model: str | None = None  # served_name; defaults to the first embedding model
    rerank_model: str | None = None  # served_name; defaults to the first rerank model
    generator_model: str | None = None  # served_name; defaults to the first lm model
    rerank: bool = True  # toggleable so the rerank lift can be measured
    top_k: int = Field(default=20, gt=0)  # dense candidates retrieved
    top_n: int = Field(default=5, gt=0)  # passed into the grounded prompt after rerank
    chunk_size: int = Field(default=220, gt=0)  # words per chunk
    chunk_overlap: int = Field(default=40, ge=0)  # words of overlap
    store_dir: str = ".crucible/rag"
    max_context_chars: int = Field(default=6000, gt=0)
    # Token budget for the grounded answer. Reasoning models (Qwen3 emits <think> blocks)
    # need headroom so the answer isn't cut off; raise this if answers truncate.
    answer_max_tokens: int = Field(default=1024, gt=0)


class Registry(_Strict):
    profile: str = "auto"
    server: ServerConfig = Field(default_factory=ServerConfig)
    models: list[ModelEntry] = Field(default_factory=list)
    profiles: dict[str, ProfileSpec] = Field(default_factory=dict)
    rag: RagConfig = Field(default_factory=RagConfig)

    @model_validator(mode="after")
    def _unique_served_names(self) -> Registry:
        names = [m.served_name for m in self.models]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate served_name(s) in registry: {sorted(dupes)}")
        if self.profile != "auto" and self.profiles and self.profile not in self.profiles:
            raise ValueError(
                f"profile '{self.profile}' is not defined in profiles: {sorted(self.profiles)}"
            )
        return self
