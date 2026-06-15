# Optional monitoring stack

Crucible's observability is **in-app and requires nothing here**: the server exposes a
native Prometheus `/metrics` endpoint and serves a live dashboard at `/observability`. The
engine runs native on the host to reach Metal; it is never containerized.

This directory is only for users who want **long-term metric retention and Grafana
dashboards**. It runs Prometheus and Grafana as CPU-only side-services in Docker, which is
acceptable because they never touch the GPU. It is entirely optional.

```
docker compose -f ops/docker-compose.yml up -d
# Prometheus: http://localhost:9090   Grafana: http://localhost:3000 (admin/admin)
```

Prometheus scrapes the host-native server at `host.docker.internal:8000/metrics`.
