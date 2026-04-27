# Mesh Router Refactor Master Plan

Status: draft for implementation
Scope: `mesh-router` only; `mesh-worker` is treated as the hardened runtime/control-plane dependency.

## Goals

- Keep MR compatible with the existing Mesh system and OpenAI-compatible clients.
- Make MW the authoritative source of runtime truth for MW-managed lanes.
- Remove known reliability hazards before structural refactors.
- Make tracked code/config/docs safe for eventual public release.
- Add repeatable MR-side certification so request routing, model loading, swaps, and streaming can be verified without ad hoc manual checks.
- Reduce `mesh_router/app.py` from a monolith into focused modules with clear boundaries.

## Current Risk Summary

- `mesh_router/app.py` is too broad: API handlers, request lifecycle, leases, swaps, MW orchestration, inventory, metrics, and compatibility paths are interleaved in one file.
- Streaming and non-streaming request execution duplicate lease, heartbeat, MW load, error, and metrics handling.
- MW command responses can currently be treated as successful while still pending.
- MR has competing sources of truth: MR DB lane rows, MW Kafka state, Redis runtime cache, direct HTTP probes, and legacy MeshBench sync.
- Direct probes can update MW-managed lane state, racing MW during model swaps.
- Legacy swap-gateway paths still coexist with MW Kafka control.
- Several tests covering MW effective readiness are disabled by renaming them to `skip_test_*`.
- Tracked defaults/docs/scripts contain homelab-specific values that should not ship in public code.
- Runtime secrets have placeholder defaults without fail-fast startup validation.

## Verified High-Priority Findings

- `mesh_router/app.py:_reroute_displaced_lease` references `downstream_model` before assignment when looking up `model_id`; this can break swap-displaced lease rerouting.
- `mesh_router/mw_control.py:send_command` returns `ok=True, pending=True` on MW response timeout. Several call sites only check `ok`, so a slow swap can be treated as ready before MW reaches a terminal state.
- `mesh_router/mw_control.py:send_command` defaults missing response payload `ok` to success. Malformed or partial MW responses should not be success by default.
- `mesh_router/probe.py:probe_once` probes and updates all non-router lanes, including MW-managed lanes.
- `mesh_router/sync.py:run_forever` is launched by default by `mesh_router/cli.py` unless `--no-sync` is passed. Its MeshBench role needs to be explicitly kept, gated, or removed.
- There are 9 disabled `skip_test_*` tests covering MW readiness/inventory/pinned-routing behavior.

## Phase 1a: Correctness Baseline

Purpose: fix known reliability bugs and prevent false-success behavior before broader changes.

Tasks:
- Fix `_reroute_displaced_lease` so model lookup uses the requested/resolved model in the correct order and cannot reference unbound variables.
- Change MW command result semantics so `pending=True` is not equivalent to ready/success for inference or swap execution call sites.
- Change malformed MW responses with missing `payload.ok` to fail closed.
- Add helper logic for MW load commands: submit command, detect pending, poll `mw_transitions`, and only proceed after terminal `completed`.
- Gate or skip direct probe updates for MW-managed lanes in `probe.py`.
- Decide and implement default behavior for `sync.py`: feature-flag it off by default if MeshBench is no longer authoritative, or document and constrain it if still needed.
- Add fail-fast startup validation for placeholder runtime secrets unless an explicit development override is set.
- Triage all `skip_test_*` tests: revive, rewrite, or delete with replacement coverage.

Success criteria:
- Full unit suite passes.
- New/updated tests cover MW pending command handling, malformed command responses, MW-managed probe exclusion, secret guard behavior, and displaced lease reroute.
- No MW-managed request opens gRPC inference until MR has confirmed the requested MW model load completed or was already loaded.
- Operator-facing API returns pending/timeout information instead of silently routing into a half-swapped lane.

Primary files:
- `mesh_router/app.py`
- `mesh_router/mw_control.py`
- `mesh_router/probe.py`
- `mesh_router/sync.py`
- `mesh_router/cli.py`
- `mesh_router/config.py`
- `tests/test_mw_control.py`
- `tests/test_streaming_mw.py`
- `tests/test_strict_mw_no_baseurl_fallback.py`
- `tests/test_mw_*`
- `tests/test_inventory_api.py`
- `tests/test_router_pin_worker.py`

## Phase 1b: Public Hygiene And Certification

Purpose: make MR safe to publish and establish repeatable live validation.

Tasks:
- Replace tracked homelab defaults with public-safe placeholders or env-only settings.
- Move private deployment values to untracked/local overlay documentation.
- Sanitize docs, scripts, Kubernetes examples, comments, and test fixtures where real machine names/IPs are not required.
- Add a public hygiene grep/check script similar to the MW process.
- Add MR live certification scripts for:
  - non-streaming chat
  - streaming chat
  - embeddings
  - image generation
  - pinned worker
  - pinned lane
  - MW command submit/pending/poll
  - inventory/runtime truth comparison
- Add certification output format with summary, per-case status, request IDs, lane IDs, worker IDs, and sanitized troubleshooting details.

Success criteria:
- Public hygiene scan has no tracked real hostnames, private IPs, private paths, or plaintext secrets outside explicitly ignored local files.
- Certification can be run against the current deployment and produces a durable report.
- Certification failures include enough context to debug from MR/MW logs.

Primary files:
- `mesh_router/config.py`
- `.env.example`
- `README.md`
- `k8s/*`
- `scripts/*`
- `ops/*`
- new `scripts/certify_mr*.py` or equivalent
- new `docs/` runbook files

## Phase 2: Request Lifecycle Extraction

Purpose: reduce monolith risk without changing behavior.

Target shape:
- `mesh_router/request_store.py`: router request create/touch/fetch/serialize/health.
- `mesh_router/lease_store.py`: lease acquire/heartbeat/release/cleanup.
- `mesh_router/request_executor.py`: shared execution lifecycle for chat, embeddings, images.
- `mesh_router/streaming.py`: SSE-specific stream relay and finalization.
- `mesh_router/perf_observe.py`: request performance observation and headers.
- `mesh_router/api_handlers.py` or small route modules: FastAPI route handlers only.

Tasks:
- Extract request persistence helpers from `app.py`.
- Extract lease helpers from `app.py`.
- Extract shared finalization logic so streaming and non-streaming paths use the same state transitions and lease cleanup.
- Centralize response header construction.
- Centralize error-to-status-code mapping.
- Preserve route behavior through tests before and after each extraction.

Success criteria:
- `app.py` is materially smaller and route-handler focused.
- Streaming and non-streaming request paths share lifecycle/finalization code.
- Existing tests pass with no live behavior regression.

## Phase 3: MW Authority Boundary

Purpose: make source-of-truth rules explicit and enforceable.

Tasks:
- Define a single runtime truth priority order:
  - fresh Redis MW runtime cache
  - MW state DB
  - MR DB static metadata
  - direct probe only for non-MW or explicitly permitted compatibility paths
- Make MW-managed lane binding explicit and repairable.
- Remove reliance on inferred `base_url` host plus default gRPC port where explicit MW endpoint metadata exists or can be ingested.
- Ensure route placement uses MW runtime readiness and actual model for MW-managed lanes.
- Ensure inventory and route resolution agree on effective lane status.
- Add repair tooling for lane metadata drift: missing `control_plane`, missing `mw_host_id`, missing `mw_lane_id`, stale backend type, stale base URL port.

Success criteria:
- MW-managed lanes cannot silently fall back to stale MR DB or direct HTTP truth.
- `/api/inventory`, `/mesh/inventory`, `/api/routes/resolve`, and `/v1/*` routing agree on lane readiness.
- Metadata repair can run in dry-run and apply modes.

Primary files:
- `mesh_router/mw_overlay.py`
- `mesh_router/runtime_state.py`
- `mesh_router/router.py`
- `mesh_router/route_resolver.py`
- `mesh_router/inventory.py`
- `mesh_router/app.py`

## Phase 4: Command And Swap Reliability

Purpose: make model load/swap behavior deterministic from MR's perspective.

Tasks:
- Create a focused MW command service that wraps Kafka command submit, timeout, pending status, polling, terminal interpretation, and diagnostics.
- Treat MW `load_model` as a state transition with progress, not a best-effort preflight.
- Ensure MR only routes inference after target lane/model readiness is confirmed.
- Isolate legacy swap-gateway code behind an explicit non-MW path.
- Add cancellation/timeout behavior for slow load commands and stale transitions.
- Surface detailed MW transition errors through MR APIs and logs.

Success criteria:
- A model swap/load has one visible MR lifecycle with request ID correlation to MW transition ID.
- Slow swaps return pending/pollable status instead of false success.
- Failed swaps include useful MR and MW diagnostics.
- Legacy swap code cannot be accidentally used for MW-managed lanes.

Primary files:
- `mesh_router/mw_control.py`
- new `mesh_router/mw_commands.py`
- new `mesh_router/swap_control.py`
- `mesh_router/app.py`
- `scripts/mw_command_smoke.py`

## Phase 5: Inventory And Model Catalog Alignment

Purpose: prevent stale/non-runnable model options from being exposed or selected.

Tasks:
- Align MR model catalog and lane viability with MW `validated_candidates`.
- Decide whether MR DB `lane_model_viability` remains authoritative for non-MW only or becomes advisory for MW lanes.
- Stop exposing model options that MW says are rejected, missing, unsupported, or not loadable on the lane.
- Add model/candidate diff endpoints or scripts for MR DB vs MW runtime truth.
- Keep model tags and selection aliases while ensuring resolved model names map to runnable artifacts.

Success criteria:
- MR does not route to models that MW cannot load on the target lane.
- Inventory clearly distinguishes current, viable, rejected, stale, remote, and unverified models.
- Model catalog repair produces deterministic changes and dry-run output.

Primary files:
- `mesh_router/inventory.py`
- `mesh_router/router.py`
- `mesh_router/route_resolver.py`
- `mesh_router/mw_overlay.py`
- `mesh_router/scanner_utils.py`
- SQL migrations as needed

## Phase 6: Observability, Correlation, And Operator UX

Purpose: make failures diagnosable without shelling into every host first.

Tasks:
- Ensure every MR request has a stable request ID propagated through response headers, MW Kafka commands, MW gRPC metadata, logs, and perf observations.
- Add structured event records for request lifecycle, MW command lifecycle, routing decisions, and final backend result.
- Expand health/readiness endpoints to include DB, MW state DB, Redis runtime cache, Kafka command path, MW consumer freshness, and deployment revision.
- Add operator-friendly APIs for current active requests, active MW transitions, lane locks, stale leases, and last error per lane.
- Add log redaction for secrets and private paths.

Success criteria:
- Given an `X-Mesh-Request-Id`, an operator can trace route selection, model load, backend execution, and final status.
- Health endpoints distinguish MR process health from dependency readiness.
- Certification reports include request IDs that map to logs and DB rows.

## Phase 7: Deployment Hardening And Open-Source Readiness

Purpose: make deployment reproducible and repository-safe.

Tasks:
- Ensure Kubernetes deployment uses digest-pinned images or an explicit safe pull policy.
- Keep private deployment overlays outside tracked public files.
- Document local development, private deployment, and public deployment separately.
- Add CI checks for tests, formatting/linting if adopted, public hygiene, and config validation.
- Remove or clearly archive obsolete workblock docs that contain private operational detail before public release.

Success criteria:
- Fresh clone can run tests and local dev with no private assumptions.
- Production deploy path is deterministic and auditable.
- Repository is safe to publish after applying documented private-file exclusions.

## Open Decisions

- Is MeshBench still an active dependency for any lane source, or should `sync.py` be disabled by default and eventually removed?
- Should legacy swap-gateway support remain for non-MW lanes, or can it be deprecated after all current workers are MW-managed?
- Should direct HTTP probes be allowed for MW-managed multimodal compatibility lanes, or should MW gRPC/proxy support be extended first?
- Should private homelab values stay in untracked local files, a private deployment repo, or sealed Kubernetes overlays only?
- What is the minimum live certification matrix for declaring MR hardened: three current workers only, or include opportunistic/Mac workers later?
