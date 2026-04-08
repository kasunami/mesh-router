# Workblock Handoff: Image Lane Activation Router Fix

Commit: `c356c9b`

Issue fixed:
- swap-driven image lane recovery could leave a lane stuck in `status=suspended`
- the recovery SQL only restored lanes from `offline`, not `suspended`

Change:
- `mesh_router/app.py`
  - `_set_lane_suspension(...)` now restores lanes when status is `offline` or `suspended`
- regression coverage:
  - `tests/test_lane_suspension.py`

Why it mattered:
- the recovered Deskix SD lane could be healthy at runtime but remain non-ready in router truth after sibling-stop/swap activity

Validation:
- `pytest tests/test_lane_suspension.py tests/test_mw_lane_readiness_overlay.py -q`
- deployed image digest:
  - `10.0.1.48:5000/mesh-router@sha256:b79f23991783cdec2b2b40eee164fef31090f11f0e25fd2f62283b3b53f1e271`
