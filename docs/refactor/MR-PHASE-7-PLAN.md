# MR Phase 7 Plan

Status: historical implementation plan; not an active status tracker
Parent plan: `MR-REFACTOR-MASTER-PLAN.md`

## Purpose

Finish deployment/open-source hardening and reduce known runtime warnings.

## Tasks

- Keep public deployment examples placeholder-safe.
- Preserve digest/unique-tag deployment guidance.
- Remove FastAPI `on_event` deprecation by using lifespan startup.
- Run full tests and public hygiene scan.
- Confirm no accidental MW repo changes remain.

## Success Criteria

- Full unit suite passes without the FastAPI `on_event` deprecation warning.
- Public hygiene scan passes.
- MW repo remains clean.
