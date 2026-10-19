# Qualified logical model catalog and dispatch

Enable `MESH_ROUTER_QUALIFIED_CATALOG_ENABLED=true` after reviewing explicit
artifact profiles. The legacy catalog remains available when disabled.

## Contract and qualification

`GET /api/models/selectable` is read-only. It returns the
`mesh-router-selectable-logical-catalog` contract, scope
`selectable_logical_models`, and a `models` array. Each entry includes `id`,
`name`, `tags`, positive `context_limit` and `output_limit`, plus separate
`availability` booleans: `present`, `loadable`, `loaded`, `suspended`, and
`dispatch_ready`. Host names, paths, URLs and lane identifiers are omitted.
`GET /v1/models` publishes these same logical IDs, including qualified unloaded
models; availability is descriptive and never a readiness probe.

A profile is an explicit `model_tuning_profiles.settings.qualified_dispatch`
object tied to a lane, model and exact local artifact. Required fields:
`qualified: true`, a quantization-preserving `public_id`, `artifact_path`,
`context_limit`, `output_limit`, `engine_context`, `reasoning_limit`,
`safety_tokens`, `parallel: 1`, and `evidence`. Optional `runtime_aliases` and
`request_aliases` are exact strings, never fuzzy family/quantization matching.
`runtime` identifies `llama` or `q27`. Qualifications must cite existing host
profiles or verified results; downloading a file does not qualify it.

The catalog combines identical explicit public IDs, preserving distinct
quantizations/packages. One concrete profile must support the advertised
context/output pair. Dispatch rechecks that pair on its chosen host, including
any tighter `lane_model_policy.max_ctx`. Legacy file-size/KV heuristics do not
qualify a profile or veto an already reviewed CPU/GPU offload profile. Cached
legacy viability results remain diagnostic and are not authoritative for this
path. Installed files without a qualified profile remain outside this catalog.

## Placement and safety

Clients submit only a model ID to `/v1/chat/completions`. Explicit legacy aliases
can be accepted but are not advertised. Host/lane pins are rejected. MR first
prefers a loaded eligible instance with sufficient budget; larger requests can
select a larger eligible peer. Unavailable/busy hosts return a retryable error.

A PostgreSQL session advisory lock reserves the entire host before loading.
A committed router lease and heartbeat cover the load and inference. Active
router leases on sibling lanes and mapped worker leases prevent reservation;
legacy router lease acquisition uses the same lock key. A final source-event
freshness/suspension/allowlist check occurs after potentially slow loading.
Model changes go through the existing MW command client and wait for completion.
Direct health/model checks separately verify actual identity and engine context;
they never refresh worker-heartbeat timestamps. Failed load or pre-dispatch
validation attempts to restore the prior model while retaining the reservation.
Failed restoration suspends that lane. A stream disconnect releases its lease
and reservation. This dispatch path does not use unauthenticated operator
lane-status routes or introduce new access grants.

Output limits count all generated tokens: hidden reasoning, answer and tool
calls. The qualified path does not apply legacy Qwen token expansion. Admission
requires formatted prompt + total completion + safety reserve <= profile context.
For llama.cpp, `/apply-template` and `/tokenize` count the formatted prompt;
for the q27 build without these endpoints, an intentionally conservative UTF-8
byte bound includes messages, tools, tool choice and response schema. Runtime
reserves and templates still need profiling before increasing budgets. A lower
request context does not resize the server's existing KV allocation.

## OpenCode

`scripts/catalog_to_opencode_patch.py` converts a saved catalog into a provider
model map containing exact logical IDs and both `limit.context` and
`limit.output`. It rejects malformed/duplicate/missing bounds and host-specific
fields; validation failure leaves existing files untouched. Merge the reviewed
model map into each client without adding placement hints. Keep default and
small-model references within that map. OpenCode versions that require paired
limits must not receive a partial limit or legacy `context_window` substitute.
Compaction must reserve output plus the configured safety allowance.

## Recovery and rollback

1. Confirm the exact running image/package revision, consumer membership and
   broker lag. Do not infer heartbeat freshness from an HTTP response.
2. Inspect `mw_hosts.last_heartbeat_event_at`. State/response/probe time is not
   heartbeat time. Apply the nullable, unbackfilled event-time column before
   code requiring it; bound lock acquisition and retain the old image.
3. Verify the worker's source clock, intended model/engine context, active leases,
   host exclusions and approved suspension reason before enabling dispatch.
4. Compare the selectable catalog with the reviewed artifact profiles, then
   exercise non-streaming, streaming, tool calls, context rejection, reservation
   contention and a reversible unloaded-model load/restore.
5. Roll back the deployment image and feature flag through its GitOps source.
   Restore client model maps from private backups. The extra nullable heartbeat
   column can remain for old code. Do not drop it while corrected code runs.
   Preserve excluded video hosts, resource budgets and unrelated worktrees.

Unit regressions cover source-time isolation, qualification, artifact IDs,
paired limits, total-token budgeting, exclusions, loaded reuse, reservation
before load, rollback, post-load context validation and stream cleanup.
Operational qualification is distinct from a full-context performance benchmark.
