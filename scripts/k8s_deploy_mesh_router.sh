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

# Patch BOTH initContainer and main container images, and bump deploy-revision annotation.
kubectl -n "${NS}" patch deploy "${DEPLOY}" --type=json -p "
[
  {\"op\":\"add\",\"path\":\"/spec/template/metadata/annotations\",\"value\":{}},
  {\"op\":\"add\",\"path\":\"/spec/template/metadata/annotations/mesh-router.openai.com~1deploy-revision\",\"value\":\"${REVISION}\"},
  {\"op\":\"replace\",\"path\":\"/spec/template/spec/initContainers/0/image\",\"value\":\"${IMAGE}\"},
  {\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/image\",\"value\":\"${IMAGE}\"}
]
"

kubectl -n "${NS}" rollout status deploy/"${DEPLOY}" --timeout=180s

echo "imageIDs:"
kubectl -n "${NS}" get pod -l app=mesh-router -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{range .status.containerStatuses[*]}{.name}{"="}{.imageID}{"\n"}{end}{end}' || true

echo "done"
