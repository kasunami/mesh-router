# Workblock Handoff: Recert Baseline 2026-04-08

## Scope

Close-the-loop recert work after the major truth-boundary fixes:

- keep MR/MW authoritative
- keep MC consuming MR truth
- finish image-lane runtime recovery
- suppress future duplicate benchmark work
- fix MW host health false negatives
- bring `tiffs-macbook` back into live MR truth without relaxing its power boundaries

## Landed code

### mesh-router

- `064c5fc` Filter inventory candidates by live backend truth
- `17bd64e` Fix legacy MLX MW overlay binding

Files:

- `mesh_router/app.py`
- `mesh_router/mw_overlay.py`
- `tests/test_inventory_api.py`
- `tests/test_mw_lane_readiness_overlay.py`

### mesh-worker

- `015cec7` Fix host health probe false negatives

Files:

- `mesh_worker/app.py`
- `tests/test_app.py`

### mesh-computer

- `54ff26e` Dedupe equivalent benchmark targets

Files:

- `app/model_discovery.py`
- `app/db/model_discovery.py`
- `tests/test_model_discovery.py`

## What was fixed

### Deskix image runtime

Root cause:

- `sd-gpu.service` was healthy on Deskix and listening on `127.0.0.1:21440`
- the routed image path depended on `worker-gateway-11440.service`
- that gateway was inactive on Deskix, unlike pupix1
- routed image requests therefore failed before reaching the healthy SD backend

Operational repair:

- restored the missing Deskix gateway-follow wiring under:
  - `/etc/systemd/system/sd-gpu.service.d/gateway-follow.conf`
- started `worker-gateway-11440.service`
- verified:
  - `http://127.0.0.1:21440/v1/models` returns `sd-cpp-local`
  - `http://127.0.0.1:11440/health` returns `ok=true`

### Legacy MLX overlay for `tiffs-macbook`

Root cause:

- MR legacy lane row used `lane_name=mlx`
- MW runtime state publishes the active lane as `lane_id=gpu` with `lane_type=mlx`
- the MR overlay inferred the wrong MW lane binding and never applied the live MW state

Code fix:

- legacy MLX rows now infer MW lane binding as `gpu`

Live result:

- MR inventory now shows `tiffs-macbook` lane:
  - `backend_type=mlx`
  - `effective_status=ready`
  - `current_model_name=/Users/kasunami/models/Falcon3-10B-Instruct-1.58bit`

### Host health probe false negative

Root cause:

- host-level MW `health_probe` treated empty combined-lane `service_id` as unhealthy

Code fix:

- host-scoped `health_probe` now evaluates only required running services

Live result:

- host probes for Deskix, pupix1, and `tiffs-macbook` return `ok=true`

### Future duplicate benchmark scheduling

Root cause:

- equivalent MW-backed targets could be scheduled twice when distinct legacy/synthetic rows mapped to the same effective host/lane/model benchmark target

Code fix:

- normalized `benchmark_target_key`
- de-dupe across pending/running jobs and within a sync run

## Tests

### mesh-router

- `uv run pytest -q tests/test_mw_lane_readiness_overlay.py tests/test_inventory_api.py tests/test_router_pin_worker.py tests/test_mw_grpc_target.py`
- result: `14 passed`

### mesh-worker

- `pytest -q tests/test_app.py`
- result: `17 passed`

### mesh-computer

- `pytest -q tests/test_model_discovery.py tests/test_routing_policy.py`
- result: `34 passed`

## Live proofs

### Deskix

- `sd-gpu.service` active and healthy on `21440`
- `worker-gateway-11440.service` active
- `curl http://127.0.0.1:11440/health` returns backend `ok=true`
- MR inventory now shows:
  - `image-gpu` `effective_status=ready`
  - `flux1-schnell-Q4_K_S` as local viable model

### pupix1

- image lane remained healthy
- existing image benchmarks succeeded and were stored

### tiffs-macbook

- direct MW host health probe returns:
  - `actual_profile=split_default`
  - MLX lane running
  - Falcon loaded
- local MLX backend `/v1/models` responds quickly
- direct local MLX chat request succeeds

## Remaining blockers

### mesh-computer readiness / DB acquisition

Current live blocker during final recert:

- `mesh-computer` pod became unready
- logs show `asyncpg` connection timeouts while acquiring DB connections
- this blocked a clean final benchmark closeout from MC during this workblock

This is not a router/runtime truth problem.

### `tiffs-macbook` inventory viability derivation

Current live state:

- MR now has the correct live MW readiness/model truth for the host
- raw host-scan ingestion was performed
- the host still needs a clean viability derivation pass so `local_viable_models` populates on the MLX lane and MC can queue the full benchmark set from MR inventory

This is now a router inventory/viability follow-up, not a host power-policy issue.

## Final status from this block

- Deskix image runtime: fixed live
- future duplicate benchmark scheduling: fixed
- non-image inventory truth: tightened
- MW health probe false negative: fixed
- `tiffs-macbook` runtime truth: fixed live
- final benchmark/rebaseline closeout: partially blocked by `mesh-computer` DB connectivity
