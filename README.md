# MeshRouter

MeshRouter is an OpenAI-compatible routing and control-plane service built for a
personal AI-operations lab. It connects orchestrated jobs to available model
workers while keeping route selection, worker state, and recovery logic outside
the model response itself.

The service is the connective tissue between a job orchestrator such as
MeshComputer, shared operational context such as MeshBrain, and heterogeneous
model-serving lanes. It accepts familiar OpenAI-style requests, resolves an
eligible lane, forwards the request, and returns route metadata that other
services can observe and verify.

This repository is a public code sample from a working lab environment. It
demonstrates production-minded operating patterns, but it is not presented as a
polished commercial product or a turnkey platform.

## Evaluator quick tour

- Start with this `README.md` for scope, capabilities, and the runnable
  evaluator path.
- Read [`ARCHITECTURE.md`](ARCHITECTURE.md) for request flow, state ownership,
  trust boundaries, and failure behavior.
- Review [`examples/`](examples/) for sanitized requests and response metadata.
- Inspect [`tests/`](tests/) for routing, readiness, integration, and failure
  coverage.
- Check [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for the automated
  verification path and [`CHANGELOG.md`](CHANGELOG.md) for the public baseline.

## What this demonstrates

- **Model-agnostic orchestration:** route work across local, cloud, CPU, GPU,
  MLX, image, and other worker lanes without coupling callers to one backend.
- **Scoped routing controls:** accept explicit worker, lane-type, and exact-lane
  hints for jobs that need deterministic placement.
- **Operational readiness:** combine configured inventory with worker-reported
  health, loaded-model state, and validated capabilities before selecting work.
- **Separation of concerns:** make routing decisions deterministically in the
  control plane rather than asking a model to choose where its own request runs.
- **Verification and observability:** expose request IDs and resolved route
  metadata, record performance observations, and provide inventory and
  certification paths for operators.
- **Failure recovery:** support bounded model-load requests, lane readiness
  checks, strict pinning, retry controls, and explicit failures instead of
  silently routing incompatible work.

## How MeshRouter fits into the larger mesh

```text
Client or scoped job
        |
        v
MeshComputer (orchestration and job state)
        |
        v
MeshRouter (route resolution and request lifecycle)
        |
        +----> MeshWorker control plane (desired/actual lane state)
        |
        +----> local or cloud model-serving lane
        |
        v
Response + route metadata + operational evidence

MeshBrain preserves shared context, handoffs, and operational history around
the workflow; it is not in the model-output decision loop.
```

MeshRouter focuses on routing and lane control. MeshComputer owns higher-level
job orchestration. MeshWorker owns host-level model lifecycle and reports actual
state. MeshBrain provides durable context and coordination for operators and
agents. Keeping these responsibilities separate makes failures easier to
classify and recovery actions easier to bound.

## Quick evaluator path

From a fresh clone with Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . pytest
python -m pytest
scripts/check_public_hygiene.sh
python scripts/certify_mr.py --mode dry-run
```

The dry-run certification command prints the live checks it would perform
without contacting a router or model worker. Live certification requires an
operator-provided deployment URL, models, and lane identifiers; run
`python scripts/certify_mr.py --help` for those options.

## Run tests

Install the package in editable mode with its test runner:

```bash
python -m pip install -e . pytest
```

Run the full unit and API test suite:

```bash
python -m pytest
```

Run the public-repository hygiene gate:

```bash
scripts/check_public_hygiene.sh
```

The tests exercise deterministic route resolution, strict lane pinning,
MeshWorker state overlays, streaming behavior, backend compatibility,
performance observations, and failure handling without requiring access to the
private lab network.

## OpenAI-compatible and operator APIs

OpenAI-compatible subset:

- `POST /v1/chat/completions` (including SSE streaming)
- `POST /v1/embeddings`
- `POST /v1/images/generations`

Operator-facing APIs:

- `GET /api/inventory` returns hosts, lanes, viable models, and effective
  MeshWorker state.
- `POST /api/routes/resolve` performs deterministic route resolution by
  explicit target or capability tags.
- `POST /api/mw/commands` submits a host/lane control-plane command.
- `GET /api/mw/commands/{request_id}` reports command progress and outcome.
- `POST /api/perf/observations` records measured lane performance.
- `GET /api/perf/expectations` returns recent performance expectations used in
  candidate ranking.

## Request routing controls

OpenAI-compatible requests may include routing hints in the JSON body or the
corresponding HTTP header:

- `mesh_pin_worker` / `x-mesh-pin-worker` prefers a specific worker.
- `mesh_pin_lane_type` / `x-mesh-pin-lane-type` prefers a lane type such as
  `cpu`, `gpu`, or `mlx`.
- `mesh_pin_lane_id` / `x-mesh-pin-lane-id` requires an exact lane and fails if
  that lane cannot serve the request; it does not silently fall back.

Successful responses include the resolved route in headers:

- `X-Mesh-Request-Id`
- `X-Mesh-Worker-Id`
- `X-Mesh-Lane-Id`
- `X-Mesh-Model-Name`

### FireCalc PDF capability routing

FireCalc PDF ingestion should use capability tags rather than a host name. A
certified visual parser model is tagged through the model-tags API with one or
both of:

- `firecalc.pdf.visual` for page/image understanding
- `firecalc.pdf.tables` for table-capable visual extraction

For example, a scheduler can preflight a visual route without knowing whether
Deskix, Pupix, or Packhub will serve it:

```json
POST /api/routes/resolve
{
  "modality": "chat",
  "tags": ["firecalc.pdf.visual"],
  "allow_opportunistic": true
}
```

Those aliases imply `requires_multimodal=true`; MeshRouter will fail closed if
only text lanes are available. The later OpenAI-compatible chat request may use
`model: "firecalc.pdf.visual"` with image content. Model-as-tag resolution then
selects a ready certified concrete model and returns its route metadata.

## MeshWorker integration

MeshRouter can use MeshWorker as a Kafka control plane and gRPC data plane for
selected lanes:

- A lane opts in through
  `lanes.proxy_auth_metadata.control_plane = "mw"`.
- Streaming chat uses MeshWorker gRPC `StreamChat` and relays OpenAI-style SSE
  chunks.
- Before streaming, MeshRouter best-effort requests the target model through
  MeshWorker's control plane.
- A background consumer ingests worker state, heartbeats, and command responses
  unless the server starts with `--no-mw-consume`.

By default, consumer state is stored in the main configured database. Set
`MESH_ROUTER_MW_STATE_DATABASE_URL` to isolate desired/actual worker state in a
separate database. Reference schemas live in `sql/012_mw_state_model.sql` and
`sql/013_mw_perf_observations.sql`.

## Performance observations

MeshRouter can record best-effort observations from real traffic. Observations
never block a response, canceled requests are dropped, and failed requests are
excluded from performance expectations.

Relevant settings:

- `MESH_ROUTER_PERF_AUTO_OBSERVE_ENABLED`
- `MESH_ROUTER_PERF_AUTO_OBSERVE_SAMPLE_RATE`
- `MESH_ROUTER_PERF_AUTO_OBSERVE_MIN_ELAPSED_MS`
- `MESH_ROUTER_PERF_AUTO_OBSERVE_MAX_TOTAL_MS`
- `MESH_ROUTER_ROUTE_DEBUG_HEADERS_ENABLED`

When debug headers are enabled and an expectation exists, responses may include
sample count, observation time, first-token latency, decode throughput, and
total latency metadata.

## Local model inventory

Install the package and scan a model root:

```bash
python -m pip install -e .
mesh-router inventory /path/to/models
```

Scan an archive model root:

```bash
mesh-router archive-inventory /path/to/archive archive-id --provider model-archive
```

The commands produce structured inventory for GGUF, SafeTensors, and MLX model
layouts along with basic host facts.

## Local configuration

Copy `.env.example` to `.env` and provide deployment-specific values outside
Git:

```bash
cp .env.example .env
```

MeshRouter loads `.env` from the current working directory. Process environment
variables take precedence, so deployment platforms can inject settings without
using a local file.

The example configuration and Kubernetes manifests contain placeholders only.
Database migrations and shared-cluster changes should be applied through the
operator's normal review and coordination process.

Inspect the currently configured hosts, lanes, and model artifacts without
starting the service:

```bash
python scripts/get_lane_info.py
```

The script reads `MESH_ROUTER_DATABASE_URL`, writes JSON to standard output,
and returns a nonzero exit status when the database query fails.

## Build

Build a container image with an explicit tag:

```bash
scripts/build_image.sh <tag>
```

Contributor setup, protobuf regeneration, package builds, and validation are
documented in `CONTRIBUTING.md`. The system boundaries and failure model are
documented in `ARCHITECTURE.md`.

## Safety and public hygiene

- No plaintext credentials, private network addresses, personal filesystem
  paths, or lab-specific hostnames belong in committed source, tests, or docs.
- Runtime secrets are supplied through environment variables or the deployment
  platform's secret manager.
- Example manifests remain placeholders. Do not commit an unsealed Kubernetes
  `Secret`.
- Exact lane pinning fails closed when the requested lane is unavailable.
- Live certification is explicit; the documented evaluator command defaults to
  dry-run and does not contact model workers.
- `scripts/check_public_hygiene.sh` scans tracked and untracked public files,
  including tests, for known private identifiers and RFC1918 addresses.

Before publishing a change, run:

```bash
python -m pytest
scripts/check_public_hygiene.sh
```
