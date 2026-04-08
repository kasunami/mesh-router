# Workblock Handoff: MR/MW Runtime Truth Hardening

Date: 2026-04-08

## Summary

This block hardened `mesh-router` so CPU text lanes on legacy MR rows can use MW as the authoritative load-model control plane instead of falling back to MR's brittle rsync auto-swap path.

Primary outcome:

- `pupix1` pinned CPU routing is fixed and proven live again for an exact GGUF request on a real llama.cpp CPU lane.
- `Static-Deskix` CPU publication is now MW-authoritative: only locally resolvable chat models are published on the router inventory path, and valid local CPU requests succeed.
- `Static-Deskix` missing models no longer fail through rsync host-key verification first; they now fail narrowly through MW viability/load-model truth.
- `image-gpu` publication truth is fixed live on both `Static-Deskix` and `pupix1`.
- One downstream caveat remains in `mesh-computer`: stale discovery rows are not pruned away even after router inventory truth is corrected.

## Root Causes

### Deskix

There were two separate Deskix truth problems:

1. Legacy CPU rows were not always treated as MW-authoritative, so missing models fell through to MR's old rsync auto-swap path.
2. `/api/inventory` was still publishing raw `remote_viable_models` for MW-authoritative lanes even after `/api/lanes/{lane_id}/capabilities` had been tightened.

That combination made Deskix look broader and more benchable than it really was.

After the fixes:

- `Static-Deskix` CPU publishes only the locally resolvable chat models:
  - `LFM2.5-350M-Q4_K_M.gguf`
  - `Qwen3.5-0.8B-Q4_K_M.gguf`
  - `Qwen3.5-2B-Q4_K_M.gguf`
  - `Qwen3.5-4B-Q4_K_M.gguf`
  - `Qwen3.5-9B-Q4_K_M.gguf`
- `remote_viable_models` is now empty on that MW-authoritative CPU lane
- a valid local Deskix CPU request now succeeds
- a missing model now fails through MW with:
  - `model is not viable for this lane`

That is the correct authority boundary: the lane can serve the valid local candidates and cleanly excludes the missing ones.

### pupix1

There were two real `pupix1` CPU issues:

1. legacy untagged CPU lanes were not always treated as MW-authoritative for load-model/routing
2. exact artifact requests such as `Qwen3.5-4B-Q4_K_M.gguf` did not match the lane's loaded alias form (`qwen3.5-4b`), so the router tried to swap an already-loaded model

Once MW authority was restored and exact-artifact requests were allowed to match the loaded alias/tags, pinned CPU routing on `pupix1` worked again.

## Code Changes

Repo: `mesh-router`

- `mesh_router/app.py`
  - `_mw_target_for_lane()` now infers MW authority for legacy CPU/GPU/combined rows using host/lane binding instead of requiring explicit `proxy_auth_metadata.control_plane="mw"`.
  - inferred MW bindings now accept either:
    - a concrete `mw_lanes` row, or
    - at least an MW host heartbeat/truth row
  - capability metadata now exposes normalized backend type
  - capability response `capabilities` now respects `sd` backend for image-style lanes instead of always defaulting to chat
  - `/api/inventory` now applies the same MW-authoritative locality filter as the per-lane capability endpoint, so remote candidates are hidden on MW-owned lanes
  - exact artifact requests now fall back to normalized lookup keys/tags when the runtime reports a loaded alias instead of the full artifact name
- `mesh_router/router.py`
  - exact artifact requests now fall back to normalized lookup keys/tags during lane selection, which prevents unnecessary swaps when a lane already has the requested model loaded under an alias
- `tests/test_mw_grpc_target.py`
  - updated inferred-MW tests for the new host/lane existence semantics
  - added a host-truth fallback test for legacy CPU rows
- `tests/test_inventory_api.py`
  - inventory regression now locks in MW-authoritative remote-candidate suppression
- `tests/test_router_pin_worker.py`
  - added regression for exact artifact request matching a loaded alias on a pinned worker lane

## Tests

Ran:

- `cd /home/kasunami/Projects/mesh-router && uv run pytest -q tests/test_inventory_api.py tests/test_router_pin_worker.py tests/test_mw_grpc_target.py tests/test_backend_compatibility.py tests/test_mw_lane_readiness_overlay.py`
  - `16 passed, 1 warning`
- `python3 -m py_compile /home/kasunami/Projects/mesh-router/mesh_router/app.py /home/kasunami/Projects/mesh-router/mesh_router/router.py /home/kasunami/Projects/mesh-router/tests/test_inventory_api.py /home/kasunami/Projects/mesh-router/tests/test_router_pin_worker.py`

## Live Validation

### pupix1 pinned CPU request

Request:

- `POST /v1/chat/completions`
- `mesh_pin_worker="pupix1"`
- `mesh_pin_lane_type="cpu"`
- model `Qwen3.5-4B-Q4_K_M.gguf`

Result:

- `HTTP 200`
- model resolved as `qwen3.5-4b`
- assistant returned `ok`

### Deskix pinned CPU request

Request:

- `POST /v1/chat/completions`
- `mesh_pin_worker="Static-Deskix"`
- `mesh_pin_lane_type="cpu"`
- model `DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf`

Old failure surface:

- rsync host-key verification failure

Valid local request:

- `POST /v1/chat/completions`
- `mesh_pin_worker="Static-Deskix"`
- `mesh_pin_lane_type="cpu"`
- model `Qwen3.5-4B-Q4_K_M.gguf`
- result: `HTTP 200`
- assistant returned `ok`

Missing-model failure surface:

- `HTTP 503`
- `no READY lanes available serving DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf and auto-swap failed: 409: model is not viable for this lane`

This proves MR is now using MW-authoritative loadability/viability truth instead of brittle MR-side copy logic for that legacy CPU lane.

### Image-lane publication

Both image lanes now publish the narrow truthful capability set:

- Deskix `image-gpu`
  - `capabilities: ["images", "inference"]`
  - `backend_type: "sd"`
  - `local: ["flux1-schnell-Q4_K_S"]`
  - `remote: []`
- pupix1 `image-gpu`
  - `capabilities: ["images", "inference"]`
  - `backend_type: "sd"`
  - `local: ["flux1-schnell-Q4_K_S.gguf"]`
  - `remote: []`

### Inventory proof

Live `/api/inventory` for `Static-Deskix` CPU now shows:

- `status: ready`
- `current_model_name: qwen3.5-4b`
- `local_viable_models`: only the five local Qwen/LFM chat models
- `remote_viable_models: []`

### In-pod router proof

On the new router image, `_mw_target_for_lane()` now resolves:

- Deskix CPU -> `MwGrpcTarget(endpoint='10.0.0.99:50061', host_id='static-deskix', lane_id='cpu')`
- pupix1 CPU -> `MwGrpcTarget(endpoint='10.0.0.95:50061', host_id='pupix1', lane_id='cpu')`

## Deployment Notes

Built and pushed:

- image tag: `10.0.1.48:5000/mesh-router:05810660edff`
- digest: `sha256:91f0510a4227b78a8f6d2eda2f5f05a98cd37f05e2728b5aaf5c5dc1d1b4cf4b`

Important deployment truth:

- the repo-local `/home/kasunami/k3s` checkout is dirty and not a safe commit target
- the clean GitOps source of truth was `/tmp/k3s-manifests-clean`, pushed as revision `f6b7d93`
- Argo briefly stayed stale on revision `932224c`; a hard refresh was required before it converged to `f6b7d93`
- final live deployment state is a single `mesh-router` pod on digest `sha256:91f0510a4227b78a8f6d2eda2f5f05a98cd37f05e2728b5aaf5c5dc1d1b4cf4b`

## Remaining Caveat

The remaining caveat is now downstream, not in MR/MW runtime publication:

- after a fresh `POST /models/discovery/sync`, `mesh-computer` still retains stale remote Deskix CPU discovery rows even though live MR `/api/inventory` now publishes `remote_viable_models: []` for that lane
- that means router runtime truth is fixed, but MC discovery persistence is not pruning no-longer-published remote candidates yet

That means:

- Deskix CPU runtime truth: fixed
- Deskix valid CPU path: fixed
- pupix1 pinned CPU runtime truth: fixed
- image-lane capability publication truth: fixed
- MR auto-swap dependency on Deskix CPU: reduced/replaced by MW-authoritative viability/loadability truth
- MC stale discovery pruning: still needs follow-up in `mesh-computer`
