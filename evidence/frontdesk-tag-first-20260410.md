# Front-Desk Tag-First Routing Evidence - 2026-04-10

## Summary

MC front-desk quick-chat now requests the host-agnostic selection tag `qwen3.5:0.8B` instead of the concrete `qwen3.5-0.8b` alias. MR resolves that tag using live lane/MW truth and constrains chat tag resolution to `llama` lanes. MW-managed non-stream chat now uses the MW gRPC data path, matching the existing streaming path, instead of direct-posting to the lane base URL.

## Commits

- `mesh-computer` `77cbdaa` - `Use tag-first frontdesk model selection`
- `mesh-router` `cdbf366` - `Resolve model selection tags through MW routing`
- `mesh-router` `a0b47df` - `Keep MW load commands model-intent based`
- `mesh-router` `f078999` - `Constrain chat tag resolution to llama lanes`
- GitOps `cfd23c0` - `Deploy mesh-router chat tag backend constraint`

## Live Deployment

Authoritative GitOps manifest:

`/home/kasunami/srv/k3s-manifests/apps/ai-tools/mesh-router/mesh-router.yaml`

Live router image:

`10.0.1.48:5000/mesh-router@sha256:41325cfed587dbbfcd655f983294737a089c2d3dd67ae63a3058857bad4e4c07`

Argo/Kubernetes proof:

```text
mesh-router-7c678557bf-wfgvv  Running  true  10.0.1.48:5000/mesh-router@sha256:41325cfed587dbbfcd655f983294737a089c2d3dd67ae63a3058857bad4e4c07
Synced  Healthy
```

## Live Functional Proof

Tag resolution:

```json
{
  "ok": true,
  "choice": {
    "lane_id": "a098640e-d7c0-40a1-b568-ca037befac58",
    "worker_id": "Static-Mobile-2",
    "base_url": "http://10.0.0.132:21434",
    "lane_type": "gpu",
    "backend_type": "llama",
    "current_model_name": "Qwen3.5-0.8B-Q4_K_M.gguf",
    "resolved_model": "qwen3.5:0.8B"
  },
  "candidates_considered": 1
}
```

Non-stream tag request:

```text
POST /v1/chat/completions model=qwen3.5:0.8B stream=false
HTTP/1.1 200 OK
X-Mesh-Lane-Id: a098640e-d7c0-40a1-b568-ca037befac58
X-Mesh-Worker-Id: Static-Mobile-2
assistant: Hello! How can I help you today?
```

CLI quick-chat:

```text
mode: quick_chat | classifier: heuristic | classifier_model: heuristic | confidence: 0.76 | model: qwen3.5:0.8B | context: none | job: no | reason: small_conversational_input
Hey! How's the day going?
```

Forced front-desk transport failure remains non-durable:

```text
MR_URL=http://127.0.0.1:9 mc hi
mode: quick_chat ... model: qwen3.5:0.8B | context: none | job: no
Front-desk reply is temporarily unavailable: frontdesk reply failed: [Errno 111] Connection refused
```

## Notes

During live validation, Static-Mobile-2 MW had all text services degraded because the user services were stopped while MW desired them running. Starting `llama-worker.service` restored the qwen GPU lane to `running`/`healthy`; MW then republished healthy lane truth and MR resolved the tag normally.

Remaining architecture backlog: embeddings/images still use their existing transport paths. The change here unifies MW-managed chat non-stream with MW gRPC semantics; broader transport unification should be handled separately.
