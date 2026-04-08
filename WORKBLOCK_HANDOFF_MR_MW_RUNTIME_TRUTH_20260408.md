# Workblock Handoff: MR/MW Runtime Truth Hardening

Date: 2026-04-08

## Summary

This block hardened `mesh-router` so CPU text lanes on legacy MR rows can use MW as the authoritative load-model control plane instead of falling back to MR's brittle rsync auto-swap path.

Primary outcome:

- `pupix1` pinned CPU routing is fixed and proven live for a real llama.cpp CPU lane request.
- `Static-Deskix` false local CPU availability no longer fails through rsync host-key verification first; it now fails truthfully through MW load-model with a local model-resolution error.
- Benchability truth is better aligned with MW runtime authority, but one publication caveat remains for image lanes.

## Root Causes

### Deskix

The false-positive Deskix CPU benchability came from router-side truth drift:

- `lane_model_viability` still contained stale "local viable" rows for artifacts that were not actually present locally.
- routing/inventory had already been tightened to require `host_model_artifacts.present=true`, but legacy CPU auto-swap still fell back to MR's rsync path because `_mw_target_for_lane()` returned `None` for untagged legacy CPU rows.
- when the route tried to auto-swap `DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf`, MR used remote copy instead of MW load-model and surfaced an rsync host-key failure.

After the fix, the same request now fails truthfully via MW:

- `llama-cpu could not resolve model 'DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf' from /etc/mesh-llama/llama-cpu.json`

That is the correct authority boundary: the model is not locally resolvable on Deskix.

### pupix1

The original `activate_profile split_llama_cpu` "stuck at started" symptom was stale interpretation, not the final blocker:

- live `mw_transitions` rows showed the transition had actually completed
- `mw_lanes` showed `pupix1 cpu` as `llama.cpp`, `running`, `healthy`

The real issue was that legacy untagged CPU lanes were not being treated as MW-authoritative for load-model/routing. Once that was corrected, pinned CPU routing on `pupix1` worked again.

## Code Changes

Repo: `mesh-router`

- `mesh_router/app.py`
  - `_mw_target_for_lane()` now infers MW authority for legacy CPU/GPU/combined rows using host/lane binding instead of requiring explicit `proxy_auth_metadata.control_plane="mw"`.
  - inferred MW bindings now accept either:
    - a concrete `mw_lanes` row, or
    - at least an MW host heartbeat/truth row
  - capability metadata now exposes normalized backend type
  - capability response `capabilities` now respects `sd` backend for image-style lanes instead of always defaulting to chat
- `tests/test_mw_grpc_target.py`
  - updated inferred-MW tests for the new host/lane existence semantics
  - added a host-truth fallback test for legacy CPU rows

## Tests

Ran:

- `cd /home/kasunami/Projects/mesh-router && uv run pytest -q tests/test_mw_grpc_target.py tests/test_backend_compatibility.py tests/test_mw_lane_readiness_overlay.py tests/test_router_pin_worker.py`
  - `13 passed, 1 warning`
- `python3 -m py_compile /home/kasunami/Projects/mesh-router/mesh_router/app.py /home/kasunami/Projects/mesh-router/tests/test_mw_grpc_target.py`

## Live Validation

### pupix1 pinned CPU request

Request:

- `POST /v1/chat/completions`
- `mesh_pin_worker="pupix1"`
- `mesh_pin_lane_type="cpu"`
- model `Qwen3.5-4B-Q4_K_M.gguf`

Result:

- `HTTP 200`
- `x-mesh-lane-id: 0a0e3d56-1f56-41be-8a29-498ba53fbbb3`
- `x-mesh-worker-id: pupix1`
- `x-mesh-model-name: Qwen3.5-4B-Q4_K_M.gguf`
- assistant returned `ok`

### Deskix pinned CPU request

Request:

- `POST /v1/chat/completions`
- `mesh_pin_worker="Static-Deskix"`
- `mesh_pin_lane_type="cpu"`
- model `DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf`

Old failure surface:

- rsync host-key verification failure

New failure surface:

- `HTTP 503`
- `no READY lanes available serving DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf and auto-swap failed: 409: llama-cpu could not resolve model 'DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf' from /etc/mesh-llama/llama-cpu.json`

This proves MR is now using MW-authoritative load-model truth instead of brittle MR-side copy logic for that legacy CPU lane.

### In-pod router proof

On the new router image, `_mw_target_for_lane()` now resolves:

- Deskix CPU -> `MwGrpcTarget(endpoint='10.0.0.99:50061', host_id='static-deskix', lane_id='cpu')`
- pupix1 CPU -> `MwGrpcTarget(endpoint='10.0.0.95:50061', host_id='pupix1', lane_id='cpu')`

## Deployment Notes

Built and pushed:

- image tag: `10.0.1.48:5000/mesh-router:20260408-123200`
- digest: `sha256:4e5b608a1e88086226a650e6caa2e639dabd0da2e24eb4ef7390e616b831a68d`

Important deployment truth:

- direct `kubectl set image` and the repo-local deploy script were not enough because Argo was reconciling from `/home/kasunami/k3s/apps/ai-tools/mesh-router/mesh-router.yaml`
- that manifest was still pinned to the old router image
- I updated the manifest locally and applied it so the running deployment could use the new digest

## Remaining Caveat

One publication caveat remains:

- `/api/lanes/{lane_id}/capabilities` for `image-gpu` lanes is still over-advertising chat-style models in some live responses
- this appears to come from image-lane aliasing against generic MW `gpu` truth and stale/stored capability state, not from the CPU routing path fixed in this block

That means:

- Deskix CPU runtime truth: fixed
- pupix1 pinned CPU runtime truth: fixed
- MR auto-swap dependency on Deskix CPU: reduced/replaced by MW-authoritative failure
- image-lane capability publication truth: still needs follow-up hardening
