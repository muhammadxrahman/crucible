"""Manual M0 smoke test: pull a tiny MLX model and confirm mlx_lm generates tokens.

Run once to satisfy the M0 acceptance criterion. Downloads weights on first run.

    uv run python scripts/smoke_generate.py
"""

from __future__ import annotations

from mlx_lm import generate, load

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
PROMPT = "In one sentence, what is Apple Silicon unified memory?"


def main() -> None:
    print(f"loading {MODEL} ...")
    model, tokenizer = load(MODEL)
    messages = [{"role": "user", "content": PROMPT}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    text = generate(model, tokenizer, prompt=prompt, max_tokens=64, verbose=True)
    assert text.strip(), "expected non-empty generation"
    print("\nOK: mlx_lm produced tokens.")


if __name__ == "__main__":
    main()
