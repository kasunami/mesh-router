# mesh-router

OpenAI-compatible router for LAN worker lanes.

## Safety

This repo is intended to be safe to push to a private GitHub repository.

- Do not commit plaintext secrets.
- Use environment variables for runtime secrets.
- For Kubernetes, commit only placeholder manifests and sealed artifacts.

## Local configuration

Copy `.env.example` to `.env` and fill in real values outside Git.

## MeshWorker (MW) integration

`mesh-router` can optionally use `mesh-worker` as a Kafka control plane + gRPC data plane for selected lanes.

Key behavior:
- MW-managed lanes are gated per-lane via `lanes.proxy_auth_metadata.control_plane = "mw"`.
- For MW-managed lanes, `/v1/chat/completions` streaming uses MW gRPC `StreamChat` and relays raw OpenAI-style SSE chunks.
- Before opening the gRPC stream, MR best-effort issues an MW `load_model` command over Kafka for the requested model.
- When running `mesh-router serve` (or `mesh-router` with serve flags), a background MW Kafka consumer thread ingests MW `state`/`heartbeats`/`responses` unless `--no-mw-consume` is set.

MW state persistence:
- By default, MW consumer writes `mw_*` rows into the same database as `database_url`.
- To keep MW state in a separate DB (recommended `ai_mesh`), set `MESH_ROUTER_MW_STATE_DATABASE_URL`.

Schema:
- MW desired/actual state tables are mirrored in `sql/012_mw_state_model.sql` (apply requires coordinated ops).

## Kubernetes secrets

The homelab cluster uses Bitnami Sealed Secrets. A safe workflow is:

1. Create a local plaintext Secret manifest from `k8s/mesh-router-secret.example.yaml`.
2. Seal it with `kubeseal` against the cluster controller.
3. Commit only the resulting sealed manifest.

Example:

```bash
cp k8s/mesh-router-secret.example.yaml /tmp/mesh-router-secret.yaml
$EDITOR /tmp/mesh-router-secret.yaml
kubeseal \
  --controller-name sealed-secrets-controller \
  --controller-namespace kube-system \
  --format yaml \
  < /tmp/mesh-router-secret.yaml \
  > k8s/mesh-router-secret.sealed.yaml
rm -f /tmp/mesh-router-secret.yaml
```

## Build

```bash
scripts/build_image.sh <tag>
```

## Local Model Inventory

To scan a local model root and output a structured inventory for ingestion:

```bash
pip install . psutil  # If not already installed
mesh-router inventory /path/to/models
```

This will output a JSON payload containing identified models (`.gguf`, `.safetensors`, MLX directories) and basic host facts.

To scan an archive model root:

```bash
mesh-router archive-inventory /path/to/archive archive-id --provider model-archive
```

## APIs

OpenAI-compatible subset:
- `POST /v1/chat/completions` (SSE streaming supported)
- `POST /v1/embeddings` (minimal)
- `POST /v1/images/generations` (minimal)

Control-plane / operator APIs:
- Inventory/capability plane:
  - `GET /api/inventory` — host → lanes → viable models (MW-managed lanes include MW-derived `effective_status` + `actual_model` overlay).
- Routing:
  - `POST /api/routes/resolve` — deterministic route resolution by explicit target or by tags (no LLM in the decision loop).
- MW command control:
  - `POST /api/mw/commands` — submit a control-plane command (returns `202 pending` when MW is still working).
  - `GET /api/mw/commands/{request_id}` — poll command status (reads `mw_transitions` via the MW state DB handle when configured).
- Performance expectations (MB-aligned, durable):
  - `POST /api/perf/observations` — ingest an observation into `mw_perf_observations` (table lives in MW state DB; apply `sql/013_mw_perf_observations.sql`).
  - `GET /api/perf/expectations` — fetch p50 expectations from recent observations.

Notes:
- `perf.host_id` is canonicalized to MW-style ids (`Worker A` → `worker-a`) at ingest to avoid silent lookup mismatches during routing.
- Tag-based route resolution ranks candidates using perf expectations when available (decode TPS / first-token latency for chat, total_ms for images); otherwise it falls back to deterministic defaults.

## Requestor Routing Controls (OpenAI-compatible)

OpenAI-compatible requests accept optional routing hints via request body *or* headers:

- `mesh_pin_worker` / header `x-mesh-pin-worker`: prefer a specific host.
- `mesh_pin_lane_type` / header `x-mesh-pin-lane-type`: prefer a lane type (`cpu|gpu|mlx|...`).
- `mesh_pin_lane_id` / header `x-mesh-pin-lane-id`: **exact lane selection**. If set, MR will either route to that lane exactly or fail (no silent fallback).

MR returns the resolved route as response headers:
- `X-Mesh-Request-Id`, `X-Mesh-Worker-Id`, `X-Mesh-Lane-Id`, `X-Mesh-Model-Name`

## Automatic Perf Observations

MR records best-effort performance observations from real traffic into `mw_perf_observations` (MW state DB) when enabled:
- `MESH_ROUTER_PERF_AUTO_OBSERVE_ENABLED` (default `true`)
- `MESH_ROUTER_PERF_AUTO_OBSERVE_SAMPLE_RATE` (default `1.0`)
- `MESH_ROUTER_PERF_AUTO_OBSERVE_MIN_ELAPSED_MS` (default `50`)
- `MESH_ROUTER_PERF_AUTO_OBSERVE_MAX_TOTAL_MS` (default `600000`)

Observations are best-effort and never block responses. Canceled requests are dropped; failed requests are recorded but excluded from expectations (`ok=false`).

### Optional perf expectation headers

For debugging/validation, MR can include perf expectation metadata as response/SSE headers:

- `MESH_ROUTER_ROUTE_DEBUG_HEADERS_ENABLED=true`

When enabled and when an expectation exists for the resolved `(host,lane,model,modality)`, MR adds:
- `X-Mesh-Perf-Sample-Count`
- `X-Mesh-Perf-Updated-At`
- `X-Mesh-Perf-FirstTokenMs-P50` (when available)
- `X-Mesh-Perf-DecodeTps-P50` (when available)
- `X-Mesh-Perf-TotalMs-P50` (when available)
