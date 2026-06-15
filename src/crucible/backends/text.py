"""MLX text backend: a thin wrapper over mlx-lm generation primitives.

Builds on mlx_lm.stream_generate and the sampler utilities rather than wrapping
mlx_lm.server, so the scheduler and batching work in M3 has room to grow.
"""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from .base import Delta, Final, GenEvent, SamplingParams


class MLXTextEngine:
    """Loads one text model and streams completions from it."""

    def __init__(self, model_path: str, served_name: str):
        self.model_path = model_path
        self.served_name = served_name
        self._model, self._tokenizer = load(model_path)

    def materialize(self) -> None:
        """Force weight allocation so resident memory can be measured honestly.

        mlx-lm loads weights lazily; evaluating the parameters realizes them.
        """
        mx.eval(self._model.parameters())

    def close(self) -> None:
        """Drop references and free cached buffers so eviction reclaims memory."""
        self._model = None
        self._tokenizer = None
        mx.clear_cache()

    def _render_prompt(self, messages: list[dict]) -> list[int]:
        return self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    def stream(self, messages: list[dict], params: SamplingParams) -> Iterator[GenEvent]:
        prompt = self._render_prompt(messages)
        sampler = make_sampler(temp=params.temperature, top_p=params.top_p)

        text = ""
        completion_tokens = 0
        prefill_tps = 0.0
        decode_tps = 0.0
        finish_reason = "length"

        for resp in stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=params.max_tokens,
            sampler=sampler,
        ):
            prefill_tps = resp.prompt_tps or prefill_tps
            decode_tps = resp.generation_tps or decode_tps
            completion_tokens = resp.generation_tokens

            piece, stop_hit = _apply_stop(resp.text, text, params.stop)
            if piece:
                text += piece
                yield Delta(piece)
            if stop_hit:
                finish_reason = "stop"
                break
            if resp.finish_reason is not None:
                finish_reason = resp.finish_reason
                break

        yield Final(
            prompt_tokens=len(prompt),
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            prefill_tps=prefill_tps,
            decode_tps=decode_tps,
        )


def _apply_stop(new_text: str, so_far: str, stops: list[str]) -> tuple[str, bool]:
    """Return the emittable delta and whether a stop sequence was reached.

    Truncates the delta at the first stop sequence found in the accumulated output.
    """
    if not new_text:
        return "", False
    if not stops:
        return new_text, False
    combined = so_far + new_text
    cut = min((combined.find(s) for s in stops if s and s in combined), default=-1)
    if cut == -1:
        return new_text, False
    emit = combined[len(so_far) : cut]
    return emit, True
