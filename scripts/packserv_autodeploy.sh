#!/usr/bin/env bash
set -euo pipefail

MESH_ROUTER_REPO_URL="${MESH_ROUTER_REPO_URL:-git@github.com:kasunami/mesh-router.git}"
K3S_MANIFESTS_REPO_URL="${K3S_MANIFESTS_REPO_URL:-ssh://kasunami@10.0.0.4/git-registry/k3s-manifests.git}"

BASE_DIR="${BASE_DIR:-/home/kasunami/srv}"
MESH_ROUTER_DIR="${MESH_ROUTER_DIR:-${BASE_DIR}/mesh-router}"
K3S_MANIFESTS_DIR="${K3S_MANIFESTS_DIR:-${BASE_DIR}/k3s-manifests}"

BRANCH="${BRANCH:-main}"
K3S_BRANCH="${K3S_BRANCH:-master}"

IMAGE_REPO="${IMAGE_REPO:-10.0.0.2:5000/mesh-router}"
MANIFEST_PATH="${MANIFEST_PATH:-apps/ai-tools/mesh-router/mesh-router.yaml}"

GIT_USER_NAME="${GIT_USER_NAME:-mesh-router-autodeploy}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-mesh-router-autodeploy@packserv}"

mkdir -p "${BASE_DIR}"

ensure_clone() {
  local url="$1"
  local dir="$2"
  local branch="$3"
  if [[ ! -d "${dir}/.git" ]]; then
    git clone --branch "${branch}" "${url}" "${dir}"
  fi
}

ensure_clone "${MESH_ROUTER_REPO_URL}" "${MESH_ROUTER_DIR}" "${BRANCH}"
ensure_clone "${K3S_MANIFESTS_REPO_URL}" "${K3S_MANIFESTS_DIR}" "${K3S_BRANCH}"

git -C "${MESH_ROUTER_DIR}" fetch origin "${BRANCH}"
git -C "${MESH_ROUTER_DIR}" checkout "${BRANCH}"
git -C "${MESH_ROUTER_DIR}" reset --hard "origin/${BRANCH}"

git -C "${K3S_MANIFESTS_DIR}" fetch origin "${K3S_BRANCH}"
git -C "${K3S_MANIFESTS_DIR}" checkout "${K3S_BRANCH}"
git -C "${K3S_MANIFESTS_DIR}" reset --hard "origin/${K3S_BRANCH}"
git -C "${K3S_MANIFESTS_DIR}" config user.name "${GIT_USER_NAME}"
git -C "${K3S_MANIFESTS_DIR}" config user.email "${GIT_USER_EMAIL}"

COMMIT_SHA="$(git -C "${MESH_ROUTER_DIR}" rev-parse --short=12 HEAD)"
IMAGE="${IMAGE_REPO}:${COMMIT_SHA}"

CURRENT_IMAGE="$(python3 - <<PY
from pathlib import Path
import re
p = Path("${K3S_MANIFESTS_DIR}/${MANIFEST_PATH}")
text = p.read_text()
m = re.search(r'image:\\s*([^\\s]+)', text)
print(m.group(1) if m else "")
PY
)"

if [[ "${CURRENT_IMAGE}" == "${IMAGE}" ]]; then
  echo "mesh-router already deployed at ${IMAGE}"
  exit 0
fi

cd "${MESH_ROUTER_DIR}"
IMAGE="${IMAGE}" ./scripts/build_image.sh "${COMMIT_SHA}"
docker push "${IMAGE}"

python3 - <<PY
from pathlib import Path
import re
p = Path("${K3S_MANIFESTS_DIR}/${MANIFEST_PATH}")
text = p.read_text()
text = re.sub(r'10\\.0\\.0\\.2:5000/mesh-router:[^\\s]+', "${IMAGE}", text)
p.write_text(text)
print("updated", p)
PY

if ! git -C "${K3S_MANIFESTS_DIR}" diff --quiet -- "${MANIFEST_PATH}"; then
  git -C "${K3S_MANIFESTS_DIR}" add "${MANIFEST_PATH}"
  git -C "${K3S_MANIFESTS_DIR}" commit -m "Deploy mesh-router ${COMMIT_SHA}"
  git -C "${K3S_MANIFESTS_DIR}" push origin "${K3S_BRANCH}"
fi

echo "deployed ${IMAGE}"
