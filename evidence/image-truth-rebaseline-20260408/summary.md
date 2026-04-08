# Evidence: Image Truth Rebaseline 2026-04-08

## Before

- MR `/api/inventory` showed:
  - `Static-Deskix image-gpu`: `effective_status=offline`, `local_viable_models=[]`
  - `pupix1 image-gpu`: `effective_status=offline`, `local_viable_models=[]`
- Router DB `host_model_artifacts` had Flux local paths on both hosts but `present=false`
- SSH on both hosts proved the Flux files actually existed on disk

## Fixes Applied

1. Root-scoped host inventory absence marking
2. Explicit-image-lane-over-inferred-gpu dedupe in MW overlay
3. Manual MR host scans for the real Flux roots
4. MW profile activation to `split_image_cpu_text`
5. MW health probes to refresh runtime state into MR
6. MC discovery sync + benchmark runner

## After

MR `/api/inventory`:

- `Static-Deskix image-gpu`
  - `backend_type=sd`
  - `effective_status=ready`
  - `current_model_name=flux1-schnell-Q4_K_S`
  - `local_viable_models=[flux1-schnell-Q4_K_S]`
- `pupix1 image-gpu`
  - `backend_type=sd`
  - `effective_status=ready`
  - `current_model_name=flux1-schnell-Q4_K_S`
  - `local_viable_models=[flux1-schnell-Q4_K_S]`
- `pupix1 gpu`
  - `backend_type=llama`
  - `effective_status=offline`
  - `readiness_reason=backend_mismatch`

MR DB:

- `host_model_artifacts.present=true` for:
  - `/home/kasunami/ai-worker/image-models/flux-schnell/flux1-schnell-Q4_K_S.gguf`
  - `/srv/ai_models/image-models/flux-schnell/flux1-schnell-Q4_K_S.gguf`

MW health probe:

- both `Static-Deskix` and `pupix1` reported:
  - `actual_profile=split_image_cpu_text`
  - `gpu-sd-image`
  - `actual_model=flux1-schnell-Q4_K_S`
  - `health_status=healthy`

MC:

- `Static-Deskix image-gpu flux1-schnell-Q4_K_S`
  - `availability_status=ready`
  - `benchmark_status=proven_runnable`
  - `capability_class=image`
  - `benchmark_kind=image_profile`
- `pupix1 image-gpu flux1-schnell-Q4_K_S`
  - same shape

Benchmark jobs in `ai_mesh.mc_model_benchmark_jobs`:

- `77` `pupix1` `image.standard` `flux1-schnell-Q4_K_S` `running`
- `78` `Static-Deskix` `image.standard` `flux1-schnell-Q4_K_S` `running`

## Clean Truth Matrix

- Deskix CPU: ready
- Deskix image: ready
- pupix CPU: ready
- pupix image: ready

## Known Remaining Nuance

- duplicate job `76` on `pupix1` raw `gpu` lane was already running before the dedupe fix landed
- future publication is corrected; this is an in-flight historical artifact, not current truth
