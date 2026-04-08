# MR/MW Runtime Truth Evidence

Date: 2026-04-08

## Tests

- `uv run pytest -q tests/test_mw_grpc_target.py tests/test_backend_compatibility.py tests/test_mw_lane_readiness_overlay.py tests/test_router_pin_worker.py`
  - `13 passed, 1 warning`
- `python3 -m py_compile mesh_router/app.py tests/test_mw_grpc_target.py`

## Code + Image

- repo commit: `0b25bc5`
- image digest: `10.0.1.48:5000/mesh-router@sha256:4e5b608a1e88086226a650e6caa2e639dabd0da2e24eb4ef7390e616b831a68d`

## Live Proofs

### Deskix CPU

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
{"detail":"no READY lanes available serving DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf and auto-swap failed: 409: llama-cpu could not resolve model 'DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf' from /etc/mesh-llama/llama-cpu.json"}
```

Interpretation:

- Deskix no longer fails through MR rsync host-key verification first.
- The failure is now MW-authoritative and truthful: the model is not resolvable on the Deskix CPU lane.

### pupix1 CPU

Request:

```bash
curl -sS -D /tmp/pupix_cpu_hdr6.txt -o /tmp/pupix_cpu_body6.json \
  -H 'Content-Type: application/json' \
  http://10.0.1.47:4010/v1/chat/completions \
  -d '{"model":"Qwen3.5-4B-Q4_K_M.gguf","messages":[{"role":"user","content":"Reply with exactly: ok"}],"max_tokens":8,"temperature":0,"mesh_pin_worker":"pupix1","mesh_pin_lane_type":"cpu"}'
```

Observed result:

- `HTTP/1.1 200 OK`
- `x-mesh-lane-id: 0a0e3d56-1f56-41be-8a29-498ba53fbbb3`
- `x-mesh-worker-id: pupix1`
- `x-mesh-model-name: Qwen3.5-4B-Q4_K_M.gguf`
- body completion: `ok`

Interpretation:

- pinned CPU routing on pupix1 is working again
- CPU lane readiness and MW load-model truth are reaching the request path successfully

### In-pod target resolution

On the new router image:

```python
_mw_target_for_lane(... deskix cpu ...) -> MwGrpcTarget(endpoint='10.0.0.99:50061', host_id='static-deskix', lane_id='cpu')
_mw_target_for_lane(... pupix1 cpu ...) -> MwGrpcTarget(endpoint='10.0.0.95:50061', host_id='pupix1', lane_id='cpu')
```

Interpretation:

- legacy CPU rows now resolve to MW targets instead of falling through to MR-only swap logic

## Caveat

- `image-gpu` capability publication still appears stale/over-broad in some live responses
- CPU routing truth was the fixed path in this block; image-lane publication still needs a separate cleanup pass
