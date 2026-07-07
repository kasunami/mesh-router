# MeshRouter examples

These sanitized examples show the public request and routing contract without
requiring access to a deployed mesh. The host, worker, lane, request ID, model,
and response content are illustrative placeholders.

Lane identifiers are deployment-specific and may be UUIDs in a live inventory.
The success and failure transcripts intentionally show different readiness
states for the same illustrative lane.

- `chat-completion-request.json` is a minimal non-streaming OpenAI-compatible
  chat request.
- `explicit-lane-pinning-request.json` adds `mesh_pin_lane_id` to require one
  exact lane. MeshRouter must not silently choose another lane.
- `route-metadata-response.txt` shows an illustrative successful HTTP response,
  including the route metadata headers added by MeshRouter.
- `fail-closed-exact-lane.txt` shows an exact-lane request failing with HTTP 409
  before dispatch when the required lane is unavailable.

Send either JSON request to an evaluator-controlled deployment:

```bash
curl --silent --show-error \
  --header 'Content-Type: application/json' \
  --data @examples/chat-completion-request.json \
  https://router.example.test/v1/chat/completions
```

Replace `https://router.example.test` with the URL of the deployment being
evaluated. Authentication, when required, is deployment-specific and is not
represented in these public examples.
