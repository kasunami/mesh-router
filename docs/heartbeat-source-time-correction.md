# Worker heartbeat source-time correction

## Finding

The worker freshness overlay must use a timestamp from a validated heartbeat
event. The legacy `mw_hosts.last_heartbeat_at` field may contain historical
consume-time values, and the Redis lane cache was also assigning state/response
processing time to that field. `/mesh/inventory` could further overwrite it
from a successful health probe. None of those observations establishes that a
worker heartbeat is fresh.

## Staged correction

- `mw_hosts.last_heartbeat_event_at` is a new, unbackfilled source-of-truth
  column. It stays NULL until a validated heartbeat is processed, so stale
  historical consume-time values fail closed.
- Heartbeat ingestion writes the validated source timestamp to that column.
- State and response snapshots carry the database event timestamp into Redis;
  their observation time remains separate and cannot refresh heartbeat age.
- Readiness overlays distrust legacy cache entries and health-probe timestamps,
  falling back to the dedicated event-time column.
- `/mesh/inventory` may report a reachable health probe, but cannot mark a lane
  ready or advance `mw_last_heartbeat_at` without a fresh heartbeat.

## Rollout requirements

This is a local staging branch only. Before a future rollout, review the code
and run the migration on both applicable databases through the normal migration
entrypoint. A host remains stale until its first valid heartbeat populates the
new column. No runtime DB migration, restart, status change, or model operation
was performed while staging this correction.

Offline verification: `python3 -m pytest -q tests` (236 passed), Python compile
check, `git diff --check`, and `scripts/check_public_hygiene.sh` passed.
