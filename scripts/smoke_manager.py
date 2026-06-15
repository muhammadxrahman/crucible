"""Manual M2 acceptance: real multi-model management on the GPU.

Registers two small text models, switches between them, forces LRU eviction with a
tight ceiling, and demonstrates single-resident behavior. Downloads weights on first run.

    uv run python scripts/smoke_manager.py
"""

from __future__ import annotations

from crucible.backends import SamplingParams
from crucible.config import Registry
from crucible.manager import ModelManager, RuntimeProfile, make_loader

_GIB = 1024**3

REG = Registry.model_validate(
    {
        "models": [
            {"path": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "type": "lm", "served_name": "a"},
            {"path": "mlx-community/Llama-3.2-1B-Instruct-4bit", "type": "lm", "served_name": "b"},
        ]
    }
)


def _gen(engine) -> str:  # noqa: ANN001
    out = ""
    for ev in engine.stream(
        [{"role": "user", "content": "Say hi in 3 words."}], SamplingParams(max_tokens=12)
    ):
        out += getattr(ev, "text", "")
    return out.strip()


def _resident(m: ModelManager) -> str:
    return f"{m.resident_models()} ({m.resident_bytes() / _GIB:.2f} GB)"


def main() -> None:
    # Tight ceiling so the two models cannot both stay resident -> forces eviction.
    multi = RuntimeProfile(
        name="pro64",
        ceiling_bytes=int(0.6 * _GIB),
        single_resident=False,
        default_context=8192,
        kv_bits=8,
        vision=True,
    )
    m = ModelManager(REG, multi, make_loader())

    print("[1] load 'a', generate")
    print("    ->", repr(_gen(m.acquire("a"))))
    print("    resident:", _resident(m))

    print("[2] switch to 'b' (ceiling forces LRU eviction of 'a')")
    print("    ->", repr(_gen(m.acquire("b"))))
    print("    resident:", _resident(m))
    assert m.resident_models() == ["b"], "expected 'a' evicted under the ceiling"

    print("[3] switch back to 'a' mid-session")
    print("    ->", repr(_gen(m.acquire("a"))))
    print("    resident:", _resident(m))
    assert m.resident_models() == ["a"]

    print("[4] single-resident profile evicts the previous model on every switch")
    single = RuntimeProfile(
        name="pro32",
        ceiling_bytes=100 * _GIB,
        single_resident=True,
        default_context=8192,
        kv_bits=4,
        vision=True,
    )
    ms = ModelManager(REG, single, make_loader())
    ms.acquire("a")
    ms.acquire("b")
    print("    resident after a->b:", ms.resident_models())
    assert ms.resident_models() == ["b"]

    print("\nOK: M2 manager switches, evicts by LRU under the ceiling, and honors single-resident.")


if __name__ == "__main__":
    main()
