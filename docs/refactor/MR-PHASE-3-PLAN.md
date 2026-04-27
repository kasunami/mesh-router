# MR Phase 3 Plan

Status: in progress
Parent plan: `MR-REFACTOR-MASTER-PLAN.md`

## Purpose

Make MW the explicit runtime authority for MW-managed lanes across routing, inventory, probes, and operator views.

## Tasks

- Centralize MW lane binding helpers so `app.py`, `router.py`, `probe.py`, and inventory code do not each infer authority differently.
- Keep direct HTTP probes disabled for explicit MW-managed lanes.
- Ensure MW state overlays expose effective service port, actual model, backend type, model candidates, and stale/hung job details consistently.
- Add or keep tests for:
  - explicit MW lanes do not fall back to direct base URL
  - MW effective readiness overrides stale MR DB lane rows
  - MW backend mismatch prevents wrong modality routing
  - MW observed stale swap/suspension suppression
- Add metadata repair tooling if bounded; otherwise leave it for Phase 5.

## Success Criteria

- Full unit suite passes.
- Public hygiene check passes.
- Route placement and inventory agree on MW effective readiness.
- Known direct-probe paths cannot overwrite explicit MW-managed lane truth.

## Out Of Scope

- Full model catalog reconciliation belongs to Phase 5.
- Full legacy swap-gateway removal belongs to Phase 4.
