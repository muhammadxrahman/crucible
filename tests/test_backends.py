"""Engine-level logic that must not regress, tested without loading a model."""

from crucible.backends import Delta, Final, SamplingParams, TextEngine
from crucible.backends.text import _apply_stop, render_chat_prompt


def test_render_chat_prompt_passes_enable_thinking() -> None:
    class Tok:
        def apply_chat_template(self, messages, add_generation_prompt=True, **kw):
            return kw  # echo the kwargs back

    out = render_chat_prompt(Tok(), [{"role": "user", "content": "hi"}], enable_thinking=False)
    assert out == {"enable_thinking": False}


def test_render_chat_prompt_falls_back_when_unsupported() -> None:
    class StrictTok:  # template that does not accept enable_thinking
        def apply_chat_template(self, messages, add_generation_prompt=True):
            return "prompt"

    out = render_chat_prompt(StrictTok(), [{"role": "user", "content": "hi"}], enable_thinking=True)
    assert out == "prompt"  # gracefully falls back, no crash


def test_apply_stop_no_stops_passes_through() -> None:
    assert _apply_stop("abc", "xy", []) == ("abc", False)


def test_apply_stop_truncates_at_sequence() -> None:
    # accumulated "Hello" + new "END now" -> emit up to END, then stop
    emit, hit = _apply_stop("END now", "Hello ", ["END"])
    assert emit == ""
    assert hit is True


def test_apply_stop_partial_then_full() -> None:
    emit, hit = _apply_stop(" in mid", "text END", ["END"])
    # stop already present in so_far boundary; nothing more to emit
    assert hit is True


def test_apply_stop_emits_prefix_before_stop() -> None:
    emit, hit = _apply_stop("keep STOP drop", "", ["STOP"])
    assert emit == "keep "
    assert hit is True


def test_sampling_params_defaults() -> None:
    p = SamplingParams()
    assert p.max_tokens == 0  # unlimited by default
    assert p.stop == []


def test_resolve_max_tokens_unlimited() -> None:
    from crucible.backends.base import UNLIMITED_MAX_TOKENS, resolve_max_tokens

    assert resolve_max_tokens(0) == UNLIMITED_MAX_TOKENS  # 0 -> unlimited
    assert resolve_max_tokens(-5) == UNLIMITED_MAX_TOKENS
    assert resolve_max_tokens(256) == 256  # a positive cap is honored


def test_fake_engine_satisfies_protocol() -> None:
    class E:
        served_name = "x"
        model_path = "y"

        def stream(self, messages, params):
            yield Delta("hi")
            yield Final(prompt_tokens=1, completion_tokens=1, finish_reason="stop")

    assert isinstance(E(), TextEngine)
