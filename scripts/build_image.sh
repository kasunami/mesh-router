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

TAG="${1:-}"
if [[ -z "${TAG}" ]]; then
  echo "usage: scripts/build_image.sh <image_tag>" >&2
  echo "example: scripts/build_image.sh 20260306h" >&2
  exit 2
fi

IMAGE="${IMAGE:-10.0.0.2:5000/mesh-router:${TAG}}"

echo "building mesh-router image"
echo "  repo_root: ${ROOT}"
echo "  image:     ${IMAGE}"

cd "${ROOT}"
docker build -t "${IMAGE}" .

echo "done"
