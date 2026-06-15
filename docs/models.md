# Models

Covers the model registry, model selection, quantization, and the hardware profiles that scale the platform across Mac memory tiers.

## The model registry

All servable models are declared in `config/models.yaml`. Source code never hardcodes a model path. This registry is the mechanism behind "run different models": one server, many models, loaded and evicted on demand.

Per-entry keys:

- `path`: a Hugging Face repo or local path (prefer `mlx-community` 4-bit converts).
- `type`: one of `lm`, `vlm`, `embedding`, `rerank`.
- `served_name`: the name clients use in the `model` field.
- `quant`: 4 or 8.
- `context_length`: max context for this model.
- `pin`: if true, never evicted by LRU.
- `ttl_seconds`: idle eviction timeout.
- `adapters` (optional): list of LoRA adapter paths for this base model.

Example:

```yaml
server:
  host: 127.0.0.1
  port: 8000
  memory_ceiling_gb: 50
  batching: true

models:
  - path: mlx-community/<qwen3-30b-a3b-4bit>
    type: lm
    served_name: primary
    quant: 4
    context_length: 32768
    pin: true

  - path: mlx-community/<qwen3-vl-30b-a3b-4bit>
    type: vlm
    served_name: vision
    quant: 4
    ttl_seconds: 600

  - path: mlx-community/<qwen3-embedding-small>
    type: embedding
    served_name: embed
    pin: true

  - path: mlx-community/<qwen3-reranker-2b>
    type: rerank
    served_name: rerank
    ttl_seconds: 1800
```

Replace the angle-bracket names with current `mlx-community` converts at build time. Specific version numbers change frequently; the model class matters more than the exact tag. Verify availability on Hugging Face before pinning.

## Selection matrix (64GB target)

| Role | Model class | Footprint (4-bit) | Expected decode | Notes |
|---|---|---|---|---|
| Daily driver, agentic | 30B-A3B MoE (Qwen3 class) | ~17GB | 50 to 90 tok/s | Default on unified memory. Strong tool calling. |
| Local coding agent | Qwen3-Coder-Next 80B-A3B | ~48GB | 10 to 15 tok/s | Fits 64GB, tight. ~71% SWE-bench Verified, 256K context, 3B active. |
| Max quality dense | Llama 3.3 70B / Qwen 72B | ~40 to 46GB | 5 to 8 tok/s | Bandwidth-limited. Offline use, sluggish interactive. |
| Vision | Qwen3-VL-30B-A3B | ~17 to 20GB | ~60 to 70 tok/s | MoE VLM: document QA, OCR, grounding. |
| Embeddings | Qwen3 Embedding (small) / all-MiniLM | <1 to 2GB | sub-ms per text | Dense retrieval vectors. |
| Reranker | Qwen3 Reranker 2B (cross-encoder) | ~1.5GB | fast | Second-stage retrieval precision. |
| Fast utility / draft | 3B to 8B dense | 2 to 5GB | 100+ tok/s | 8-bit below 3B; 4-bit degrades small models. |

Do not target 122B+ models on 64GB; they need ~70GB at 4-bit and belong on a 128GB machine.

## Quantization policy

- Default to 4-bit MLX converts. On 7B and larger, quality is within roughly 1 to 2% of bf16 on standard benchmarks.
- Use 8-bit for models under 3B, where 4-bit visibly degrades.
- Convert a model with `mlx_lm.convert --hf-path <repo> --q-bits 4 --mlx-path <out>`.
- Quantize the KV cache (4-bit) for long-context work to cut cache memory 60 to 75% at a small quality cost.

## Hardware profiles

A profile is the mechanism behind "runs on lower-spec Macs." It is a named overlay on the registry that bundles the limits that change per memory tier: which model roles to load, single vs multi-resident, default context length, KV-cache bits, and whether vision is enabled.

### Tier reference

| Total memory | Model budget | Default model | Vision | Resident models |
|---|---|---|---|---|
| 16GB | ~6 to 8GB | 7 to 8B dense (4-bit) or 3 to 4B (8-bit) | off | single |
| 24GB | ~14 to 16GB | 12 to 14B dense or small MoE | light | single |
| 32GB | ~22 to 24GB | 30B-A3B MoE (4-bit) | small VLM | single to two |
| 48GB | ~36 to 38GB | 30B MoE + embed + rerank | yes | two |
| 64GB | ~48 to 52GB | 30B MoE + vision + embed + rerank | yes | several |
| 128GB | ~104 to 112GB | up to ~122B MoE | yes | several |

### Selection logic

Resolution order at startup:

1. An explicit `--profile` flag.
2. `server.profile` in config.
3. Auto-detect from `hw.memsize`: pick the highest profile whose model budget fits.

If the declared registry exceeds the active profile's budget, warn (do not crash) and load only what fits, smallest-first for the pinned roles.

### Profile config

```yaml
profile: auto   # auto | air16 | base24 | pro32 | pro48 | pro64 | max128

profiles:
  air16:                       # 16GB Macs
    single_resident: true
    default_context: 8192
    kv_bits: 4
    vision: false
    roles:
      primary: mlx-community/<llama-3.3-8b-4bit>
      embed: mlx-community/<all-minilm-l6-v2-8bit>
  pro64:                       # 64GB M5 Pro
    single_resident: false
    default_context: 32768
    kv_bits: 8
    vision: true
    roles:
      primary: mlx-community/<qwen3-30b-a3b-4bit>
      vision: mlx-community/<qwen3-vl-30b-a3b-4bit>
      embed: mlx-community/<qwen3-embedding-small>
      rerank: mlx-community/<qwen3-reranker-2b>
```

The web UI reads the active profile and renders only the supported views (for example, it hides the vision view when `vision: false`). See `ui.md`.
