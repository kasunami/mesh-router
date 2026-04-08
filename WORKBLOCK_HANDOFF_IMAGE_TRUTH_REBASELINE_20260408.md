# Workblock Handoff: Image Truth And Rebaseline

Date: 2026-04-08

## Summary

This block fixed the remaining MR/MW image-lane truth gap and used the corrected truth to advance the rebaseline flow.

What changed:

- MR host inventory scans now mark `host_model_artifacts.present=false` only within the scanned root, instead of clobbering unrelated local roots on the same host.
- MR MW overlay now keeps explicit synthetic `image-gpu` rows authoritative and suppresses inferred legacy `gpu` rows when they share the same MW binding but disagree on backend.
- Flux is once again published as a real local viable model on Deskix and pupix `image-gpu` lanes.
- MC now sees both image lanes as `availability_status=ready` from MR truth and image benchmark jobs are running from that corrected surface.

## Exact Root Causes

### 1. Empty image viable-model publication

Root cause:

- `mesh-router` inventory ingestion was marking all local artifacts for a host absent whenever any one local root was rescanned.
- Flux lived in dedicated image roots:
  - Deskix: `/home/kasunami/ai-worker/image-models/flux-schnell`
  - pupix1: `/srv/ai_models/image-models/flux-schnell`
- later scans of other local roots flipped those Flux artifacts to `present=false`
- MR therefore had no local viable image artifact to publish, even though the file existed on disk

Fix:

- scope absence marking to the scanned `root_path` only

### 2. Duplicate pupix image publication

Root cause:

- `pupix1` had both:
  - an explicit synthetic `image-gpu` row bound to MW `gpu`
  - an inferred legacy `gpu` row for the same MW lane
- when the host entered image mode, the inferred `gpu` row inherited the real MW backend and became image-capable too
- MC then saw both lanes and queued duplicate Flux image benchmarks

Fix:

- when an inferred row shares an MW binding with an explicit MW-managed row and the backends disagree, the inferred row is forced `effective_status=offline` with `readiness_reason=backend_mismatch`
- this leaves the canonical synthetic `image-gpu` row as the only active image lane

## Files Changed

- `mesh_router/app.py`
- `mesh_router/mw_overlay.py`
- `tests/test_inventory_host_scan.py`
- `tests/test_mw_image_lane_overlay.py`
- `WORKBLOCK_HANDOFF_IMAGE_TRUTH_REBASELINE_20260408.md`
- `evidence/image-truth-rebaseline-20260408/summary.md`

## Validation

Tests:

- `cd /home/kasunami/Projects/mesh-router && uv run pytest -q tests/test_inventory_api.py tests/test_mw_lane_readiness_overlay.py tests/test_mw_image_lane_overlay.py tests/test_inventory_host_scan.py tests/test_router_pin_worker.py tests/test_mw_grpc_target.py`
- result: `14 passed, 1 warning`
- after dedupe patch:
  - `cd /home/kasunami/Projects/mesh-router && uv run pytest -q tests/test_mw_image_lane_overlay.py tests/test_inventory_host_scan.py tests/test_inventory_api.py tests/test_router_pin_worker.py tests/test_mw_grpc_target.py`
  - result: `12 passed, 1 warning`

Live proof:

- Flux file exists on both hosts and is now marked `present=true` in MR DB
- MR `/api/inventory` now shows:
  - `Static-Deskix image-gpu`: `effective_status=ready`, local viable `flux1-schnell-Q4_K_S`
  - `pupix1 image-gpu`: `effective_status=ready`, local viable `flux1-schnell-Q4_K_S`
  - `pupix1 gpu`: `effective_status=offline`, `readiness_reason=backend_mismatch`
- MW `health_probe` proved both hosts are actually in `split_image_cpu_text` and running `sd-gpu`
- MC sync now sees image rows as ready/proven-runnable on both hosts
- image benchmark jobs:
  - `77` pupix1 `image-gpu` Flux `image.standard` running
  - `78` Static-Deskix `image-gpu` Flux `image.standard` running

## Deployment

Router images built/pushed:

- `10.0.1.48:5000/mesh-router@sha256:0ceb84b15df1a31b8e1e31a00153fde1277177aaa4723eb45669a158f6f252d3`
- `10.0.1.48:5000/mesh-router@sha256:f1278d6ea952571fbf8564730268afc43bd690bd50c6654eb1592e964369234a`

GitOps:

- clean repo used: `/tmp/k3s-manifests-clean`
- commits:
  - `686cd93` `Update mesh-router image for image truth fix`
  - `46e6180` `Update mesh-router image for image lane dedupe`
  - rebased and pushed on top of remote `7e8a256` to final head `07eb94b`

Live deployment:

- `mesh-router` deployment now references `sha256:f1278d6ea952571fbf8564730268afc43bd690bd50c6654eb1592e964369234a`

## Current Rebaseline Status

- Deskix CPU: ready, valid local CPU candidates remain active
- Deskix image: ready, Flux image benchmark running
- pupix CPU: ready, valid CPU truth preserved
- pupix image: ready, Flux image benchmark running

## Remaining Caveats

- One duplicate pupix Flux benchmark job (`id=76` on the raw `gpu` lane) was already running before the dedupe overlay shipped. Future routing/publication is corrected, but that already-started job still exists in MC history.
- MR still publishes broad local model lists on some non-image lanes because viability storage is more permissive than benchmark eligibility. The image path is corrected and narrow, but broader lane-model viability cleanup is a separate follow-up.
- MW `health_probe` returns `ok=false` because the combined lane publishes an empty `service_id`; runtime state in the payload is still correct. That is a separate MW command-quality issue.
