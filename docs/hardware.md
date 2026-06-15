# Hardware

The architecture is shaped by the physics of Apple Silicon inference. These constraints explain several of the hard rules in `CLAUDE.md`.

## Primary target: M5 Pro

| Property | Value | Consequence |
|---|---|---|
| GPU cores | 20, each with a Neural Accelerator | Matrix-multiply units reached through Metal 4 TensorOps. Drives prefill and time-to-first-token. |
| Peak AI compute | ~4x M4 Pro | Prefill is compute-bound, so long prompts process fast. |
| Unified memory | 64GB | GPU-addressable. This is the model budget. |
| Memory bandwidth | 307GB/s | Decode is bandwidth-bound. This is the generation speed ceiling. |
| Neural Engine | 16-core | Used by system ML; not the main lever for LLM serving. |

## The bandwidth rule

Decode speed has a hard ceiling of roughly `bandwidth / bytes_read_per_token`. At 4-bit:

- A 70B dense model reads the full ~40GB of weights per token. 307 / 40 is about 7.7 tok/s in theory, so expect 5 to 8 tok/s in practice. Usable for offline jobs, sluggish for chat.
- A 30B Mixture-of-Experts model with 3B active reads only about 2GB per token. 307 / 2 is about 150 tok/s in theory, so expect 50 to 90 tok/s in practice. Feels like a cloud API.

Same disk footprint, an order of magnitude apart in decode speed. The M5 Pro at 307GB/s is bandwidth-constrained for large dense models. The response is to default to MoE models, not to assume a faster machine. The Neural Accelerators keep prefill fast regardless, so time-to-first-token is good across the board; the bandwidth ceiling bites only on dense-model decode.

Implication for metrics: always report prefill throughput and decode throughput separately. A single blended number hides the actual behavior of this hardware.

## Memory budgeting

Model budget is total unified memory minus a reserve for macOS and the working set.

- macOS: 3 to 4GB.
- Working set (IDE, browser, other apps, the server process): 8GB on 16 to 24GB machines, 10 to 12GB above that.

On 64GB, the practical model budget is roughly 48 to 52GB. Budget conservatively, because other processes move underneath the server at runtime.

Large models may need the GPU wired-memory ceiling raised: `sudo sysctl iogpu.wired_limit_mb=<value>`. Do not set this blindly. Starving macOS of wired memory causes swap thrash. Document any value used and leave macOS adequate headroom.

## The Docker rule

Docker Desktop on macOS runs containers inside a Linux VM. That VM has no access to the Metal GPU or the Neural Accelerators. Running the inference engine in a container drops it to CPU and removes the reason to use this hardware.

Therefore: the inference engine runs native on the host, managed by `launchd`. Docker is acceptable only for stateless CPU side-services (an optional external Prometheus or Grafana, an optional vector database). State this explicitly in the README, because a reviewer familiar with Linux and NVIDIA backends will assume the opposite.

Because the end goal is a self-contained installable app, the default install adds no external daemons at all. Observability is in-app: the gateway exposes a native Prometheus-format `/metrics` endpoint and serves an `/observability` dashboard backed by an in-memory ring buffer. External Prometheus and Grafana (Homebrew or Docker) remain available for long-term retention, but they are an optional add-on, not part of the default path.

## Scaling across Macs

The platform must run on machines from 16GB to 128GB. The codebase stays the same; the model registry and a few limits change, encoded as hardware profiles. The full tier table and the profile mechanism are in `models.md`. The short version:

- Below 32GB: one resident model at a time, no LRU juggling.
- Under 3B parameters: use 8-bit rather than 4-bit.
- Small tiers: shorter default context and a 4-bit KV cache to keep the cache small; a tiny VLM or no vision on 16GB; the smallest embedding model on 16 to 24GB.

Bandwidth, not generation number, predicts decode speed. A higher-bandwidth older chip can decode faster than a newer chip with less bandwidth. When reasoning about expected speed on a given Mac, look up that chip's bandwidth rather than assuming newer is faster.

## Sources

- Apple M5 Pro newsroom and MacBook Pro tech specs (20-core GPU, 64GB, 307GB/s, per-core Neural Accelerators).
- Apple ML Research, "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU" (prefill and decode speedups, the compute-bound vs bandwidth-bound split).
- Independent 2026 Apple Silicon benchmarks (insiderllm, jmlab, codersera, jaredwatkins) for per-tier model fit and the MoE-on-unified-memory pattern.

Verify current figures before relying on them; hardware and model details move quickly.
