# Live Runtime Truth Cache Follow-Through TODOs

Date: 2026-04-15

## Goal

Make MR route/status/inventory decisions from MW-reported live runtime truth, not stale Postgres lane fields. Keep Postgres for durable identity, policy, and audit history.

## Loose TODOs

1. Centralize current live lane reads behind one MR effective-state path.
   - First slice: make `/mesh/inventory` apply MW effective state before returning current model/status.
   - Outcome: MC-facing inventory stops exposing stale `lanes.current_model_name` when MW has fresher truth.

2. Define the runtime snapshot contract.
   - Host snapshot: host id, profile, generation, heartbeat, health.
   - Lane snapshot: lane id/type, service id, backend, actual model, desired model, actual state, health, endpoint, ETA fields, error.
   - Swap response snapshot: full affected-lane post-attempt state for success and failure.

3. Add a Redis-backed MR live-state store.
   - Use existing Redis infrastructure rather than Postgres for volatile loaded-model state.
   - TTL live keys so missing/stale cache becomes `unknown/offline`, never stale ready.
   - Keep durable transition/swap events in Postgres.

4. Update MW state ingestion to write the live-state store.
   - On MW startup/contact, write full host/lane/service snapshot.
   - On heartbeat/state change, refresh the cache.
   - On swap success/failure, update cache with actual post-attempt state.

5. Move MR read paths to the same effective-state abstraction.
   - `/mesh/inventory`
   - `/api/inventory`
   - `/api/lanes/{lane_id}/lease-status`
   - route resolution / placement
   - swap preflight and readiness checks

6. Stop treating `lanes.current_model_name` as runtime truth.
   - Keep or rename it as `last_observed_model_name` only if useful for diagnostics.
   - Do not use it for active routing when MW-managed live state is fresh.

7. Fix or bypass gateway model-list probing for MW-managed lanes.
   - Current worker gateway health-shaped `/api/tags` and `/v1/models` responses can prevent MR probes from learning loaded models.
   - With MW snapshots, router probing should be secondary diagnostics, not the source of truth.

8. Add failure-mode tests.
   - Fresh MW snapshot overrides stale DB model/status.
   - Missing/expired MW snapshot makes MW-managed lane `unknown/offline`.
   - Swap failure records actual loaded model/dead lane state.
   - Stale combined-lane cache cannot report ready when service is down.

## Progress 2026-04-15

- Implemented the first Redis live-state slice in MR. MW state snapshots now write TTL-bound host/service/lane keys under `mr:mw:*`, and the MW effective-state overlay reads Redis before falling back to Postgres.
- Kept Postgres as durable audit/fallback storage for MW hosts, lanes, services, transitions, and transition events.
- Added tests proving cache population preserves model/backend/ETA/service port truth and that fresh cache data overrides stale DB rows in effective routing/inventory state.
- Remaining cache follow-through: write post-swap success/failure snapshots from MW responses, move lease-status/swap preflight reads onto the same cache-backed abstraction, and stop treating durable lane model columns as current truth for MW-managed lanes.
## Progress 2026-04-15 Response Snapshots

- Added response-side cache refresh. MR now extracts `result.host_state` from MW command responses and writes it to Redis as `mw_response_snapshot`, including failed/rejected command outcomes when MW reports the post-attempt host state.
- This closes the first swap convergence gap: successful or failed MW commands can update live lane/backend/model truth immediately, without waiting for the next periodic state message.
- Remaining follow-through: ensure lease-status and swap preflight exclusively consume the cache-backed effective-state abstraction, then retire durable `lanes.current_model_name` as runtime truth for MW-managed lanes.

