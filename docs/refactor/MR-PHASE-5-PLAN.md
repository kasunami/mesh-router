# MR Phase 5 Plan

Status: in progress
Parent plan: `MR-REFACTOR-MASTER-PLAN.md`

## Purpose

Align MR inventory/model catalog behavior with MW's hardened model discovery so MR does not expose stale or non-runnable model options.

## Tasks

- Tighten MR archive/local scanner heuristics to avoid treating support files as runnable models.
- Preserve MLX directory detection as directory-level model detection.
- Keep MR-side catalog behavior conservative for ambiguous single files.
- Add tests for scanner filtering.
- Leave full MR DB vs MW `validated_candidates` reconciliation tooling for a later dedicated repair pass if it requires schema/API changes.

## Success Criteria

- Full unit suite passes.
- Scanner tests verify common support files are not exposed as artifacts.
- Inventory remains compatible with existing host/archive scan APIs.
