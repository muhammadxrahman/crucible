"""Render benchmark results as a Markdown report (dependency-light, no charts needed).

Keeps prefill and decode in separate columns so the bandwidth-ceiling effect on dense
decode versus MoE decode is visible (docs/hardware.md).
"""

from __future__ import annotations

from .harness import CaseResult


def _bar(value: float, peak: float, width: int = 24) -> str:
    filled = 0 if peak <= 0 else round(width * value / peak)
    return "#" * filled + "-" * (width - filled)


def to_markdown(results: list[CaseResult], meta: dict) -> str:
    lines: list[str] = []
    lines.append("# Crucible benchmark report")
    lines.append("")
    lines.append(f"- prompt: `{meta.get('prompt', '')}`")
    lines.append(f"- max_tokens: {meta.get('max_tokens', '')}")
    lines.append(f"- profile: {meta.get('profile', 'n/a')}")
    lines.append(f"- generated: {meta.get('generated', '')}")
    lines.append("")
    lines.append("Prefill and decode are reported separately; a single blended number hides")
    lines.append("the bandwidth-bound behavior of decode on Apple Silicon.")
    lines.append("")

    models = []
    for r in results:
        if r.model not in models:
            models.append(r.model)

    for model in models:
        rows = [r for r in results if r.model == model]
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            "| concurrency | prompt tok | prefill tok/s | decode tok/s | TTFT ms | agg decode |"
        )
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| {r.concurrency} | {r.prompt_tokens} | {r.prefill_tps:.1f} | "
                f"{r.decode_tps:.1f} | {r.ttft_ms:.0f} | {r.agg_decode_tps:.1f} |"
            )
        lines.append("")
        peak = max((r.agg_decode_tps for r in rows), default=0.0)
        lines.append("Aggregate decode throughput vs concurrency:")
        lines.append("")
        lines.append("```")
        for r in rows:
            lines.append(
                f"N={r.concurrency:<3} {_bar(r.agg_decode_tps, peak)} {r.agg_decode_tps:.0f}"
            )
        lines.append("```")
        lines.append("")

    return "\n".join(lines)
