# MR Phase 4 Plan

Status: in progress
Parent plan: `MR-REFACTOR-MASTER-PLAN.md`

## Purpose

Make MW command and swap behavior deterministic from MR's perspective.

## Tasks

- Extract MW command submit/wait/poll helpers into a focused module.
- Ensure `pending=True` MW command responses are never treated as ready.
- Ensure malformed MW responses fail closed.
- Keep legacy swap-gateway behavior isolated to non-MW fallback paths.
- Preserve MR API behavior for operator-submitted MW commands: pending commands remain pollable.

## Success Criteria

- Full unit suite passes.
- MW load commands used before gRPC inference wait for terminal `completed`.
- Failed or timed-out MW commands produce actionable errors.
- Operator command API still returns `202` for pending commands with a polling URL.
