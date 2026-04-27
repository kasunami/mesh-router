# MR Phase 2 Plan

Status: in progress
Parent plan: `MR-REFACTOR-MASTER-PLAN.md`

## Purpose

Reduce `mesh_router/app.py` monolith risk by extracting request lifecycle pieces into focused modules while preserving behavior.

## Scope

Phase 2 is behavior-preserving. It should not redesign routing policy, MW command semantics, inventory truth, or deployment behavior. Those are handled by later phases.

## Tasks

- Extract router lease persistence:
  - cleanup expired leases
  - list active leases
  - acquire lease
  - heartbeat lease
  - release lease
- Extract router request persistence:
  - cleanup expired requests
  - create request
  - touch request
  - cancellation checks
  - fetch request
  - request serialization/health helpers if practical
- Extract shared response/error helpers where low-risk.
- Keep FastAPI route handlers in `app.py` for now.
- Add or preserve tests around lease/request lifecycle behavior.

## Success Criteria

- Full unit suite passes after each extraction batch.
- `app.py` loses persistence/lifecycle helper bodies without behavior change.
- Imports remain acyclic and simple.
- No live deployment assumptions are changed.

## Audit Rules

- If extraction exposes a correctness bug, fix it immediately if bounded.
- If extraction exposes a larger architectural issue, add it to the master plan under Phase 3 or later.
