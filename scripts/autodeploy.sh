#!/usr/bin/env bash
set -euo pipefail

MESH_ROUTER_REPO_URL="${MESH_ROUTER_REPO_URL:-ssh://<user>@<your-git-server>/mesh-router.git}"
K3S_MANIFESTS_REPO_URL="${K3S_MANIFESTS_REPO_URL:-ssh://<user>@<your-git-server>/git-registry/k3s-manifests.git}"

BASE_DIR="${BASE_DIR:-/srv/mesh}"
MESH_ROUTER_DIR="${MESH_ROUTER_DIR:-${BASE_DIR}/mesh-router}"
K3S_MANIFESTS_DIR="${K3S_MANIFESTS_DIR:-${BASE_DIR}/k3s-manifests}"

BRANCH="${BRANCH:-main}"
K3S_BRANCH="${K3S_BRANCH:-master}"

IMAGE_REPO="${IMAGE_REPO:-registry.example/mesh-router}"
MANIFEST_PATH="${MANIFEST_PATH:-apps/ai-tools/mesh-router/mesh-router.yaml}"

GIT_USER_NAME="${GIT_USER_NAME:-mesh-router-autodeploy}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-mesh-router-autodeploy@example.invalid}"

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
IMAGE_TAG="${IMAGE_REPO}:${COMMIT_SHA}"

CURRENT_IMAGE="$(python3 - <<PY
from pathlib import Path
import re
p = Path("${K3S_MANIFESTS_DIR}/${MANIFEST_PATH}")
text = p.read_text()
m = re.search(r'image:\\s*([^\\s]+)', text)
print(m.group(1) if m else "")
PY
)"

if [[ "${CURRENT_IMAGE}" == "${IMAGE_TAG}" ]]; then
  echo "mesh-router already deployed at ${IMAGE_TAG}"
  exit 0
fi

cd "${MESH_ROUTER_DIR}"
IMAGE="${IMAGE_TAG}" ./scripts/build_image.sh "${COMMIT_SHA}"
docker push "${IMAGE_TAG}"

IMAGE_DIGEST="$(docker image inspect --format '{{index .RepoDigests 0}}' "${IMAGE_TAG}" 2>/dev/null || true)"
if [[ -z "${IMAGE_DIGEST}" ]]; then
  # Some docker configurations don't populate RepoDigests until after a pull.
  docker pull "${IMAGE_TAG}" >/dev/null
  IMAGE_DIGEST="$(docker image inspect --format '{{index .RepoDigests 0}}' "${IMAGE_TAG}" 2>/dev/null || true)"
fi
if [[ -z "${IMAGE_DIGEST}" ]]; then
  echo "fatal: could not determine pushed image digest for ${IMAGE_TAG}" >&2
  exit 2
fi

python3 - <<PY
from pathlib import Path
import re
p = Path("${K3S_MANIFESTS_DIR}/${MANIFEST_PATH}")
text = p.read_text()
text = re.sub(
    r"(^\\s*image:\\s*)(\\S*mesh-router\\S*)\\s*$",
    lambda match: f"{match.group(1)}${IMAGE_DIGEST}",
    text,
    flags=re.M,
)
text = re.sub(r"(^\\s*imagePullPolicy:\\s*)Never\\s*$", r"\\1IfNotPresent", text, flags=re.M)
p.write_text(text)
print("updated", p)
PY

if ! git -C "${K3S_MANIFESTS_DIR}" diff --quiet -- "${MANIFEST_PATH}"; then
  git -C "${K3S_MANIFESTS_DIR}" add "${MANIFEST_PATH}"
  git -C "${K3S_MANIFESTS_DIR}" commit -m "Deploy mesh-router ${COMMIT_SHA} (${IMAGE_DIGEST})"
  git -C "${K3S_MANIFESTS_DIR}" push origin "${K3S_BRANCH}"
fi

echo "deployed ${IMAGE_TAG}"
echo "  digest: ${IMAGE_DIGEST}"
