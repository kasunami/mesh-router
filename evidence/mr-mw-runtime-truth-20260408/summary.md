# MR/MW Runtime Truth Evidence

Date: 2026-04-08

## Tests

- `uv run pytest -q tests/test_inventory_api.py tests/test_router_pin_worker.py tests/test_mw_grpc_target.py tests/test_backend_compatibility.py tests/test_mw_lane_readiness_overlay.py`
  - `16 passed, 1 warning`
- `python3 -m py_compile mesh_router/app.py mesh_router/router.py tests/test_inventory_api.py tests/test_router_pin_worker.py`

## Code + Image

- repo commits:
  - `0b25bc5`
  - `091bb27`
  - `96f29f6`
  - `0581066`
- image digest: `10.0.1.48:5000/mesh-router@sha256:91f0510a4227b78a8f6d2eda2f5f05a98cd37f05e2728b5aaf5c5dc1d1b4cf4b`
- GitOps manifest revision: `f6b7d93`

## Live Proofs

### Deskix CPU inventory truth

Live `/api/inventory` now returns:

```json
{
  "host_name": "Static-Deskix",
  "lane_name": "cpu",
  "status": "ready",
  "current_model_name": "qwen3.5-4b",
  "local": [
    "LFM2.5-350M-Q4_K_M.gguf",
    "Qwen3.5-0.8B-Q4_K_M.gguf",
    "Qwen3.5-2B-Q4_K_M.gguf",
    "Qwen3.5-4B-Q4_K_M.gguf",
    "Qwen3.5-9B-Q4_K_M.gguf"
  ],
  "remote": []
}
```

Interpretation:

- Deskix CPU no longer publishes remote candidates like DeepSeek on the router inventory path.
- Valid CPU candidates are now only the locally resolvable chat models.

### Deskix CPU valid request

Request:

```bash
curl -sS -D /tmp/deskix_ok_hdr.txt -o /tmp/deskix_ok_body.json \
  -H 'Content-Type: application/json' \
  http://10.0.1.47:4010/v1/chat/completions \
  -d '{"model":"Qwen3.5-4B-Q4_K_M.gguf","messages":[{"role":"user","content":"Reply with exactly: ok"}],"max_tokens":8,"temperature":0,"mesh_pin_worker":"Static-Deskix","mesh_pin_lane_type":"cpu"}'
```

Observed result:

- `HTTP/1.1 200 OK`
- body completion: `ok`

Interpretation:

- Deskix is now actually usable for its valid local CPU candidates, not merely "honestly failing".

### Deskix CPU missing model

Request:

```bash
curl -sS -D /tmp/deskix_hdr5.txt -o /tmp/deskix_body5.json \
  -H 'Content-Type: application/json' \
  http://10.0.1.47:4010/v1/chat/completions \
  -d '{"model":"DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf","messages":[{"role":"user","content":"Reply with exactly: ok"}],"max_tokens":8,"temperature":0,"mesh_pin_worker":"Static-Deskix","mesh_pin_lane_type":"cpu"}'
```

Observed result:

- `HTTP/1.1 503 Service Unavailable`
- body:

```json
{"detail":"no READY lanes available serving DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf and auto-swap failed: 409: model is not viable for this lane"}
```

Interpretation:

- Deskix no longer fails through MR rsync host-key verification first.
- The failure is now MW-authoritative and truthful: the missing model is not viable on that lane.

### pupix1 CPU

Request:

```bash
curl -sS -D /tmp/pupix_cpu_hdr6.txt -o /tmp/pupix_cpu_body6.json \
  -H 'Content-Type: application/json' \
  http://10.0.1.47:4010/v1/chat/completions \
  -d '{"model":"Qwen3.5-4B-Q4_K_M.gguf","messages":[{"role":"user","content":"Reply with exactly: ok"}],"max_tokens":8,"temperature":0,"mesh_pin_worker":"pupix1","mesh_pin_lane_type":"cpu"}'
```

Observed result after the final alias-matching fix:

- `HTTP/1.1 200 OK`
- body `model: "qwen3.5-4b"`
- body completion: `ok`

Interpretation:

- exact artifact requests no longer trigger a bogus swap just because the runtime reports a loaded alias
- pinned CPU routing on pupix1 is working again

### In-pod target resolution

On the new router image:

```python
_mw_target_for_lane(... deskix cpu ...) -> MwGrpcTarget(endpoint='10.0.0.99:50061', host_id='static-deskix', lane_id='cpu')
_mw_target_for_lane(... pupix1 cpu ...) -> MwGrpcTarget(endpoint='10.0.0.95:50061', host_id='pupix1', lane_id='cpu')
```

Interpretation:

- legacy CPU rows now resolve to MW targets instead of falling through to MR-only swap logic

### Image lanes

Deskix:

```json
{
  "capabilities": ["images", "inference"],
  "current_model": "flux1-schnell-Q4_K_S",
  "local": ["flux1-schnell-Q4_K_S"],
  "remote": []
}
```

Pupix:

```json
{
  "capabilities": ["images", "inference"],
  "current_model": "flux1-schnell-Q4_K_S.gguf",
  "local": ["flux1-schnell-Q4_K_S.gguf"],
  "remote": []
}
```

Interpretation:

- image publication truth is fixed live on both hosts
- the router no longer over-publishes text/chat capability on those image lanes

## Caveat

- router runtime truth is fixed, but `mesh-computer` discovery still retains stale remote Deskix CPU rows after sync
- that remaining issue is downstream stale-row persistence in MC, not live MR/MW publication truth
