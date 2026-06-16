"""BatchedTextEngine: presents the TextEngine interface, backed by the scheduler.

Drop-in for MLXTextEngine. The gateway calls .stream() exactly the same way; concurrency
is handled by the shared scheduler behind it.
"""

from __future__ import annotations

from collections.abc import Iterator

from crucible.backends.base import Final, GenEvent, SamplingParams
from crucible.backends.text import render_chat_prompt

from .scheduler import BatchScheduler


class BatchedTextEngine:
    def __init__(self, scheduler: BatchScheduler, tokenizer, served_name: str, model_path: str):
        self._scheduler = scheduler
        self._tok = tokenizer
        self.served_name = served_name
        self.model_path = model_path

    def stream(self, messages: list[dict], params: SamplingParams) -> Iterator[GenEvent]:
        tokens = render_chat_prompt(self._tok, messages, enable_thinking=params.enable_thinking)
        channel = self._scheduler.submit(list(tokens), params)
        try:
            while True:
                event = channel.get()
                yield event
                if isinstance(event, Final):
                    return
        except GeneratorExit:
            # The client disconnected mid-stream: stop generating and free the KV slot instead
            # of running to EOS for output nobody will read.
            self._scheduler.cancel(channel)
            raise

    def materialize(self) -> None:
        # The model is already realized when the scheduler is built.
        pass

    def close(self) -> None:
        self._scheduler.stop()

    def snapshot(self) -> dict:
        return self._scheduler.snapshot()

    def stats(self) -> dict:
        s = self._scheduler.snapshot()
        s["kv_cache_bytes"] = self._scheduler.kv_nbytes()
        return s
