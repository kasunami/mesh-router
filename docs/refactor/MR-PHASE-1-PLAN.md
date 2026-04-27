# MR Phase 1 Plan

Status: in progress
Parent plan: `MR-REFACTOR-MASTER-PLAN.md`

## Purpose

Phase 1 establishes a safe baseline before structural refactors. It is split into:

- Phase 1a: correctness baseline
- Phase 1b: public hygiene and certification

## Phase 1a: Correctness Baseline

Tasks:
- Fix swap-displaced lease rerouting so it cannot reference unbound variables and reports clear failures.
- Make MW command timeout/malformed-response behavior fail closed for inference/swap call sites.
- Add a reusable MW load helper that waits for pending transitions before opening gRPC or declaring swaps complete.
- Prevent direct probe loops from updating MW-managed lane status/model truth.
- Gate MeshBench sync behind an explicit setting so it cannot silently mutate lane rows by default.
- Add startup validation for placeholder runtime secrets unless development secrets are explicitly allowed.
- Revive or replace disabled `skip_test_*` coverage around MW readiness and routing.

Success criteria:
- Unit tests pass.
- New/updated tests cover the correctness changes.
- MW-managed inference does not proceed from a pending `load_model`.
- Direct probes do not race MW-managed lane state.

## Phase 1b: Public Hygiene And Certification

Tasks:
- Replace tracked homelab defaults in docs/scripts/examples with public-safe placeholders.
- Add a repeatable hygiene scan for tracked files.
- Add configurable MR certification tooling for chat, streaming, embeddings, image, pinning, MW command polling, and inventory checks.
- Ensure certification artifacts are operator-useful and sanitized.

Success criteria:
- Hygiene scan passes for tracked public files.
- Certification tooling has safe defaults and can be pointed at a private MR deployment via CLI/env.
- Syntax/tests/checks pass.

## Parallel Work Split

- Main thread: production correctness changes and full integration audit.
- Worker A: revive/rewrite disabled MW-boundary tests.
- Worker B: public hygiene/docs/scripts pass.
- Worker C: certification tooling.

## Audit Rules

- If a newly discovered issue blocks Phase 1 correctness, fix it immediately.
- If a newly discovered issue is larger than Phase 1, add it to the master plan under the correct phase.
- Do not introduce private hostnames, IPs, paths, or secrets into tracked files.
