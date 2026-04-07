# mesh-router Readiness Visibility Handoff

Date: 2026-04-07

## Scope

This repo was not the primary source of the false `503 no ready lane` bug. The
behavioral root cause lived in `mesh-computer` host-target pin handling.

The work here hardened operator truth and removed a readiness-view mismatch.

## What changed

Files:
- [`mw_overlay.py`](/home/kasunami/Projects/mesh-router/mesh_router/mw_overlay.py)
- [`app.py`](/home/kasunami/Projects/mesh-router/mesh_router/app.py)
- [`schemas.py`](/home/kasunami/Projects/mesh-router/mesh_router/schemas.py)
- [`test_mw_lane_readiness_overlay.py`](/home/kasunami/Projects/mesh-router/tests/test_mw_lane_readiness_overlay.py)
- [`test_inventory_api.py`](/home/kasunami/Projects/mesh-router/tests/test_inventory_api.py)

Behavior:
- `/api/lanes` now uses the shared MW effective-status overlay
- `/api/lanes` and `/api/inventory` now expose `readiness_reason`
- stale or excluded MW-managed lanes now have machine-readable explanations

Reasons exposed:
- `mw_state_missing`
- `mw_state_unavailable`
- `stale_heartbeat`
- `not_running`
- `unhealthy`

## Validation

- router tests passed after the change
- live rollout succeeded
- `Static-Mobile-2` now appears with explicit `readiness_reason: null` on ready lanes
- the updated surfaces are available at:
  - `http://10.0.1.47:4010/api/lanes`
  - `http://10.0.1.47:4010/api/inventory`

## Commit

- branch: `main`
- commit: `c329d7431b610ef846ce43a6712baedb735ba31b`

## Remaining risk

This repo change improves diagnosis and consistency, but the false `503` bug itself
was fixed in `mesh-computer`. Any future readiness failure should now be easier to
classify as:
- no matching lane
- lane exists but is excluded
- MW state is stale/unhealthy/unavailable
