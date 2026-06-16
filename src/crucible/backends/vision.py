"""MLX vision-language backend over mlx-vlm.

Runs all MLX work on a single owned thread, which keeps arrays thread-affine and gives the
conservative VLM concurrency the roadmap calls for (VLM batching is limited). Images are
materialized once and cached by content hash across turns, so a multi-turn conversation
over one image does not re-fetch or re-decode it; `cached_tokens` from mlx-vlm is surfaced
for models that additionally reuse KV state.
"""

from __future__ import annotations

import queue
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

from crucible.manager.memory import MlxMemory

from .base import Delta, Final, GenEvent, SamplingParams, resolve_max_tokens
from .images import ImageRef, materialize, text_messages
from .loopguard import LoopGuard
from .text import _apply_stop


class MLXVLMEngine:
    type = "vlm"

    def __init__(
        self, model_path: str, served_name: str, mem: MlxMemory | None = None, cache_size: int = 20
    ):
        self.model_path = model_path
        self.served_name = served_name
        self._mem = mem or MlxMemory()
        self._cache_size = cache_size
        self._img_cache: OrderedDict[str, str] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._cached_tokens = 0
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"vlm-{served_name}")
        self.nbytes = self._ex.submit(self._load).result()

    def _load(self) -> int:
        import mlx.core as mx
        import mlx_vlm

        self._mem.clear_cache()
        before = self._mem.active_bytes()
        self._model, self._proc = mlx_vlm.load(self.model_path)
        self._cfg = self._model.config
        mx.eval(self._model.parameters())
        return max(self._mem.active_bytes() - before, 0)

    def stream_vision(
        self, messages: list[dict], params: SamplingParams, images: list[ImageRef]
    ) -> Iterator[GenEvent]:
        channel: queue.Queue = queue.Queue()
        self._ex.submit(self._run, messages, params, images, channel)
        while True:
            event = channel.get()
            yield event
            if isinstance(event, Final):
                return

    def _run(self, messages, params, images, channel: queue.Queue) -> None:
        try:
            import mlx_vlm

            paths = [self._image_path(ref) for ref in images]
            try:
                prompt = mlx_vlm.apply_chat_template(
                    self._proc,
                    self._cfg,
                    text_messages(messages),
                    num_images=len(paths),
                    enable_thinking=params.enable_thinking,
                )
            except TypeError:  # processor template doesn't take the flag
                prompt = mlx_vlm.apply_chat_template(
                    self._proc, self._cfg, text_messages(messages), num_images=len(paths)
                )
            image_arg = paths[0] if len(paths) == 1 else (paths or None)

            acc = ""
            comp = prompt_tokens = 0
            prefill_tps = decode_tps = 0.0
            finish = "length"
            penalty = params.repetition_penalty if params.repetition_penalty > 1.0 else None
            guard = LoopGuard() if params.loop_guard else None
            for r in mlx_vlm.stream_generate(
                self._model,
                self._proc,
                prompt,
                image=image_arg,
                max_tokens=resolve_max_tokens(params.max_tokens),
                temperature=params.temperature,
                top_p=params.top_p,
                repetition_penalty=penalty,
                repetition_context_size=params.repetition_context_size,
            ):
                prompt_tokens = r.prompt_tokens or prompt_tokens
                prefill_tps = r.prompt_tps or prefill_tps
                decode_tps = r.generation_tps or decode_tps
                comp = r.generation_tokens
                self._cached_tokens += r.cached_tokens or 0
                piece, stopped = _apply_stop(r.text, acc, params.stop)
                if piece:
                    acc += piece
                    channel.put(Delta(piece))
                if stopped:
                    finish = "stop"
                    break
                if r.finish_reason is not None:
                    finish = r.finish_reason
                    break
                if guard is not None and r.token is not None and guard.feed(r.token):
                    finish = "stop"
                    break

            channel.put(
                Final(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=comp,
                    finish_reason=finish,
                    prefill_tps=prefill_tps,
                    decode_tps=decode_tps,
                )
            )
        except Exception:  # noqa: BLE001 - never hang the caller on a worker error
            channel.put(Final(prompt_tokens=0, completion_tokens=0, finish_reason="error"))

    def _image_path(self, ref: ImageRef) -> str:
        if ref.sha in self._img_cache:
            self._hits += 1
            self._img_cache.move_to_end(ref.sha)
            return self._img_cache[ref.sha]
        self._misses += 1
        path = materialize(ref)
        self._img_cache[ref.sha] = path
        while len(self._img_cache) > self._cache_size:
            self._img_cache.popitem(last=False)
        return path

    def stats(self) -> dict:
        return {
            "vision_cache_hits": self._hits,
            "vision_cache_misses": self._misses,
            "vision_cached_tokens": self._cached_tokens,
        }

    def close(self) -> None:
        def _free() -> None:
            import mlx.core as mx

            self._model = None
            self._proc = None
            mx.clear_cache()

        self._ex.submit(_free).result()
        self._ex.shutdown(wait=False)
