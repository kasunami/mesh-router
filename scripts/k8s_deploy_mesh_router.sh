#!/usr/bin/env bash
set -euo pipefail

NS="${NS:-ai-tools}"
DEPLOY="${DEPLOY:-mesh-router}"
IMAGE="${IMAGE:-}"
REVISION="${REVISION:-}"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "fatal: kubectl not found" >&2
  exit 2
fi

if [[ -z "${IMAGE}" ]]; then
  echo "usage: NS=ai-tools DEPLOY=mesh-router IMAGE=<repo@sha256:digest> REVISION=<string> $0" >&2
  exit 2
fi

if [[ "${IMAGE}" != *"@sha256:"* ]]; then
  echo "fatal: IMAGE must be digest-pinned (repo@sha256:...)" >&2
  echo "got: ${IMAGE}" >&2
  exit 2
fi

if [[ -z "${REVISION}" ]]; then
  # Force a rollout even if an operator repeats the same image ref accidentally.
  REVISION="$(date +%Y%m%d-%H%M%S)"
fi

echo "deploying mesh-router"
echo "  ns:       ${NS}"
echo "  deploy:   ${DEPLOY}"
echo "  image:    ${IMAGE}"
echo "  revision: ${REVISION}"

DEPLOY_JSON="$(kubectl -n "${NS}" get deploy "${DEPLOY}" -o json)"
export DEPLOY_JSON IMAGE REVISION

# Patch container images (all containers + any initContainers) and bump deploy-revision annotation.
PATCH="$(python3 - <<'PY'
import json, os

deploy=json.loads(os.environ["DEPLOY_JSON"])
image=os.environ["IMAGE"]
revision=os.environ["REVISION"]

patch=[
  {"op":"add","path":"/spec/template/metadata/annotations","value":{}},
  {"op":"add","path":"/spec/template/metadata/annotations/mesh-router.openai.com~1deploy-revision","value":revision},
]

spec=((deploy.get("spec") or {}).get("template") or {}).get("spec") or {}
for i, _c in enumerate(spec.get("containers") or []):
  patch.append({"op":"replace","path":f"/spec/template/spec/containers/{i}/image","value":image})
for i, _c in enumerate(spec.get("initContainers") or []):
  patch.append({"op":"replace","path":f"/spec/template/spec/initContainers/{i}/image","value":image})

print(json.dumps(patch))
PY
)"

kubectl -n "${NS}" patch deploy "${DEPLOY}" --type=json -p "${PATCH}"

kubectl -n "${NS}" rollout status deploy/"${DEPLOY}" --timeout=180s

echo "imageIDs:"
kubectl -n "${NS}" get pod -l app=mesh-router -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{range .status.containerStatuses[*]}{.name}{"="}{.imageID}{"\n"}{end}{end}' || true

echo "done"
