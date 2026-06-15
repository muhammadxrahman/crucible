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


class ServerConfig(_Strict):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, gt=0, lt=65536)
    memory_ceiling_gb: float | None = Field(default=None, gt=0)
    batching: bool = True


class ProfileSpec(_Strict):
    single_resident: bool
    default_context: int = Field(gt=0)
    kv_bits: Quant
    vision: bool
    roles: dict[str, str] = Field(default_factory=dict)


class Registry(_Strict):
    profile: str = "auto"
    server: ServerConfig = Field(default_factory=ServerConfig)
    models: list[ModelEntry] = Field(default_factory=list)
    profiles: dict[str, ProfileSpec] = Field(default_factory=dict)

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
