#!/usr/bin/env bash
set -euo pipefail

# Guardrail: people sometimes end up with nested copies of this repo (e.g. .../mesh-router/mesh-router).
# Building from the wrong directory produces images that don't match the code you think you deployed.

if ! command -v git >/dev/null 2>&1; then
  echo "fatal: git not found (required to verify repo root)" >&2
  exit 2
fi

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${ROOT}" ]]; then
  echo "fatal: not inside a git repo" >&2
  exit 2
fi

if [[ "$(basename "${ROOT}")" != "mesh-router" ]]; then
  echo "fatal: expected repo root basename 'mesh-router' but got: ${ROOT}" >&2
  echo "hint: cd to the real mesh-router repo root and retry" >&2
  exit 2
fi

if ! git -C "${ROOT}" diff --quiet || ! git -C "${ROOT}" diff --cached --quiet; then
  echo "fatal: git working tree is dirty; refusing to build an image from uncommitted changes" >&2
  echo "hint: commit your changes first so the image tag uniquely maps to code" >&2
  exit 2
fi

TAG="${1:-}"
if [[ -z "${TAG}" ]]; then
  echo "usage: scripts/build_image.sh <image_tag>" >&2
  echo "example: scripts/build_image.sh 20260306h" >&2
  exit 2
fi

# Override with IMAGE or IMAGE_REPO for your registry.
IMAGE_REPO="${IMAGE_REPO:-registry.example/mesh-router}"
IMAGE="${IMAGE:-${IMAGE_REPO}:${TAG}}"

echo "building mesh-router image"
echo "  repo_root: ${ROOT}"
echo "  image:     ${IMAGE}"

cd "${ROOT}"
NO_CACHE="${NO_CACHE:-0}"
EXTRA_BUILD_FLAGS=()
if [[ "${NO_CACHE}" == "1" ]]; then
  EXTRA_BUILD_FLAGS+=(--no-cache)
fi

# Deploy hygiene:
# - `--pull` reduces the risk of building on stale base layers.
# - Prefer unique tags (git SHA) or digests in k8s.
docker build --pull --platform linux/amd64 -t "${IMAGE}" "${EXTRA_BUILD_FLAGS[@]}" .

echo "done"
