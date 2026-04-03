#!/usr/bin/env bash
set -euo pipefail

NS="${NS:-ai-tools}"
DEPLOY="${DEPLOY:-mesh-router}"

IMAGE="${IMAGE:-}"
REVISION="${REVISION:-}"

usage() {
  cat >&2 <<EOF
usage:
  NS=ai-tools DEPLOY=mesh-router IMAGE=<image-ref> REVISION=<string> scripts/k8s_deploy_mesh_router.sh

notes:
  - Prefer digest refs: 10.0.1.48:5000/mesh-router@sha256:...
  - If deploying by tag, this script forces:
      - imagePullPolicy: Always
      - a rollout via template annotation bump
EOF
}

if [[ -z "${IMAGE}" || -z "${REVISION}" ]]; then
  usage
  exit 2
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "fatal: kubectl not found" >&2
  exit 2
fi

echo "deploying ${NS}/${DEPLOY}"
echo "  image:    ${IMAGE}"
echo "  revision: ${REVISION}"

# Patch both initContainers and containers.
kubectl -n "${NS}" patch deploy "${DEPLOY}" --type='json' -p "[
  {\"op\":\"add\",\"path\":\"/spec/template/metadata/annotations\",\"value\":{}},
  {\"op\":\"add\",\"path\":\"/spec/template/metadata/annotations/mesh-router.openai.com~1deploy-revision\",\"value\":\"${REVISION}\"},
  {\"op\":\"replace\",\"path\":\"/spec/template/spec/initContainers/0/image\",\"value\":\"${IMAGE}\"},
  {\"op\":\"replace\",\"path\":\"/spec/template/spec/initContainers/0/imagePullPolicy\",\"value\":\"Always\"},
  {\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/image\",\"value\":\"${IMAGE}\"},
  {\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/imagePullPolicy\",\"value\":\"Always\"}
]"

echo "waiting for rollout..."
kubectl -n "${NS}" rollout status deploy "${DEPLOY}" --timeout=180s

echo "verifying deploy hygiene..."
NS="${NS}" DEPLOY="${DEPLOY}" "$(dirname "$0")/k8s_verify_deploy_hygiene.sh"

echo "done"

