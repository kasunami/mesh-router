# MR Phase 6 Plan

Status: in progress
Parent plan: `MR-REFACTOR-MASTER-PLAN.md`

## Purpose

Improve operator visibility and request correlation without introducing new infrastructure dependencies.

## Tasks

- Add a dependency/status endpoint that reports MR DB, MW state DB, Redis runtime cache configuration, MW control enablement, and deployment revision.
- Preserve existing lightweight readiness/liveness endpoints for Kubernetes.
- Ensure request IDs remain surfaced in OpenAI-compatible response headers and certification output.
- Track larger tracing/log redaction/event-stream work for future hardening if it requires broader schema changes.

## Success Criteria

- Full unit suite passes.
- Health endpoint is safe to call without private data exposure.
- Existing `/healthz`, `/health/liveliness`, and `/health/readiness` compatibility remains intact.
