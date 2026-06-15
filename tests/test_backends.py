"""Engine-level logic that must not regress, tested without loading a model."""

from crucible.backends import Delta, Final, SamplingParams, TextEngine
from crucible.backends.text import _apply_stop


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
    assert p.max_tokens == 512
    assert p.stop == []


def test_fake_engine_satisfies_protocol() -> None:
    class E:
        served_name = "x"
        model_path = "y"

        def stream(self, messages, params):
            yield Delta("hi")
            yield Final(prompt_tokens=1, completion_tokens=1, finish_reason="stop")

    assert isinstance(E(), TextEngine)
