# Workblock Handoff (MR routing/control-plane)

Date: 2026-04-03

## What changed

- Added optional, feature-flagged requestor-facing **perf expectation headers** for OpenAI-compatible responses and chat SSE:
  - `MESH_ROUTER_ROUTE_DEBUG_HEADERS_ENABLED=true`
  - Emits `X-Mesh-Perf-*` headers when an expectation exists for the resolved `(host,lane,model,modality)`.

## Why

Routing + perf registry is already live, but consumers/operators still lacked an easy, compatibility-safe way to confirm what MR expects for a route without separate API calls. This improves debuggability and makes the control-plane more requestor-grade while keeping defaults safe (off).

## Files changed

- `Projects/mesh-router/mesh_router/config.py` (adds `route_debug_headers_enabled`)
- `Projects/mesh-router/mesh_router/app.py` (adds `_maybe_add_perf_expectation_headers` + wires into `/v1/chat/completions`)
- `Projects/mesh-router/tests/test_perf_expectation_headers.py` (new unit tests)
- `Projects/mesh-router/README.md` (docs for env var + emitted headers)

## Tests run

```bash
cd Projects/mesh-router
python -m unittest discover -s tests -p 'test_*.py'
```

## Notes / safety

- Header emission is **best-effort** and does not affect routing decisions.
- When disabled (default), MR does not perform the expectation lookup for request paths.
- If the MW perf table is missing or the MW state DB is unreachable, requests still succeed; headers simply will not be present.

