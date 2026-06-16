"""MLX text backend: a thin wrapper over mlx-lm generation primitives.

Builds on mlx_lm.stream_generate and the sampler utilities rather than wrapping
mlx_lm.server, so the scheduler and batching work in M3 has room to grow.
"""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from .base import Delta, Final, GenEvent, SamplingParams
from .loopguard import LoopGuard


class MLXTextEngine:
    """Loads one text model and streams completions from it (non-batched path).

    Used directly when batching is off, and as the reproducible-sampling path even when
    batching is on (batched mode shares one RNG lane). See docs/roadmap.md M3.
    """

    def __init__(
        self, model_path: str, served_name: str, *, model=None, tokenizer=None, prefix_cache=None
    ):
        self.model_path = model_path
        self.served_name = served_name
        self._prefix = prefix_cache  # optional PrefixCache for shared-prefix prefill reuse
        if model is not None and tokenizer is not None:
            self._model, self._tokenizer = model, tokenizer
        else:
            self._model, self._tokenizer = load(model_path)

    def materialize(self) -> None:
        """Force weight allocation so resident memory can be measured honestly.

        mlx-lm loads weights lazily; evaluating the parameters realizes them.
        """
        mx.eval(self._model.parameters())

    def stats(self) -> dict:
        if self._prefix is None:
            return {}
        s = self._prefix.stats.snapshot()
        return {"prefix_hits": s["hits"], "prefix_misses": s["misses"]}

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
        cache, feed = self._seed_cache(prompt)

        text = ""
        completion_tokens = 0
        prefill_tps = 0.0
        decode_tps = 0.0
        finish_reason = "length"

        kwargs = {"prompt_cache": cache} if cache is not None else {}
        processors = _logits_processors(params)
        if processors:
            kwargs["logits_processors"] = processors
        guard = LoopGuard() if params.loop_guard else None
        for resp in stream_generate(
            self._model,
            self._tokenizer,
            feed,
            max_tokens=params.max_tokens,
            sampler=sampler,
            **kwargs,
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
            if guard is not None and guard.feed(resp.token):  # runaway repetition
                finish_reason = "stop"
                break

        if cache is not None:
            self._store_prefix(prompt, cache, completion_tokens)

        yield Final(
            prompt_tokens=len(prompt),  # always the full prompt, even when a prefix was reused
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            prefill_tps=prefill_tps,
            decode_tps=decode_tps,
        )

    # --- prefix caching (only when a PrefixCache was supplied) ---

    def _seed_cache(self, prompt: list[int]) -> tuple[object | None, list[int]]:
        """Return (cache, tokens_to_prefill). Reuses a cached prefix when one matches."""
        if self._prefix is None:
            return None, prompt
        seed, matched = self._prefix.lookup(prompt)
        cache = make_prompt_cache(self._model)
        if seed is not None:
            for layer, state in zip(cache, seed, strict=False):
                layer.state = state
            return cache, prompt[matched:]
        return cache, prompt

    def _store_prefix(self, prompt: list[int], cache, completion_tokens: int) -> None:
        """Snapshot the full-prompt KV (trim the generated tail) for future reuse."""
        try:
            if completion_tokens > 0 and can_trim_prompt_cache(cache):
                trim_prompt_cache(cache, completion_tokens)
            state = [tuple(mx.array(a) for a in layer.state) for layer in cache]
            self._prefix.store(prompt, state)
        except Exception:
            pass  # caching is best-effort; never fail a request over it


def _logits_processors(params: SamplingParams):
    """Build the repetition-penalty logits processors, or None when disabled (penalty <= 1).

    The penalty is what keeps weak/quantized models from collapsing into verbatim loops and
    failing to emit EOS.
    """
    if params.repetition_penalty and params.repetition_penalty > 1.0:
        return make_logits_processors(
            repetition_penalty=params.repetition_penalty,
            repetition_context_size=params.repetition_context_size,
        )
    return None


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
