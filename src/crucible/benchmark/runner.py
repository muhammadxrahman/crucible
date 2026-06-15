"""Load a benchmark spec, run it against real models in-process, write a Markdown report.

A spec is YAML:

    prompt: "Write three sentences about Apple Silicon."
    max_tokens: 64
    concurrency: [1, 2, 4]
    models:
      - mlx-community/Qwen2.5-0.5B-Instruct-4bit
      - mlx-community/Llama-3.2-1B-Instruct-4bit
"""

from __future__ import annotations

import time
from pathlib import Path

import yaml

from crucible.config import ModelEntry
from crucible.manager import make_loader

from .harness import run_case
from .report import to_markdown

DEFAULT_REPORT_DIR = Path("benchmarks/reports")


def run_spec(spec_path: str | Path, out_dir: str | Path = DEFAULT_REPORT_DIR) -> Path:
    spec = yaml.safe_load(Path(spec_path).read_text())
    prompt = spec["prompt"]
    max_tokens = int(spec["max_tokens"])
    concurrency = list(spec["concurrency"])
    messages = [{"role": "user", "content": prompt}]

    loader = make_loader(batching=True)
    results = []
    for path in spec["models"]:
        entry = ModelEntry(path=path, type="lm", served_name=path.split("/")[-1])
        engine, _ = loader(entry)
        try:
            for n in concurrency:
                res = run_case(engine, messages, max_tokens, n)
                res.model = entry.served_name
                results.append(res)
        finally:
            close = getattr(engine, "close", None)
            if callable(close):
                close()

    meta = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    md = to_markdown(results, meta)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = out / f"bench-{time.strftime('%Y%m%d-%H%M%S')}.md"
    report.write_text(md)
    return report
