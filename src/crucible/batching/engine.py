"""BatchedTextEngine: presents the TextEngine interface, backed by the scheduler.

Drop-in for MLXTextEngine. The gateway calls .stream() exactly the same way; concurrency
is handled by the shared scheduler behind it.
"""

from __future__ import annotations

from collections.abc import Iterator

from crucible.backends.base import Final, GenEvent, SamplingParams

from .scheduler import BatchScheduler


class BatchedTextEngine:
    def __init__(self, scheduler: BatchScheduler, tokenizer, served_name: str, model_path: str):
        self._scheduler = scheduler
        self._tok = tokenizer
        self.served_name = served_name
        self.model_path = model_path

    def stream(self, messages: list[dict], params: SamplingParams) -> Iterator[GenEvent]:
        tokens = self._tok.apply_chat_template(messages, add_generation_prompt=True)
        channel = self._scheduler.submit(list(tokens), params)
        while True:
            event = channel.get()
            yield event
            if isinstance(event, Final):
                return

    def materialize(self) -> None:
        # The model is already realized when the scheduler is built.
        pass

    def close(self) -> None:
        self._scheduler.stop()

    def snapshot(self) -> dict:
        return self._scheduler.snapshot()
