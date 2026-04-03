#!/usr/bin/env bash
set -euo pipefail

NS="${NS:-ai-tools}"
DEPLOY="${DEPLOY:-mesh-router}"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "fatal: kubectl not found" >&2
  exit 2
fi

yaml="$(kubectl -n "${NS}" get deploy "${DEPLOY}" -o yaml)"
image_lines="$(printf '%s\n' "${yaml}" | rg -n '^\s*image:\s' || true)"
pull_lines="$(printf '%s\n' "${yaml}" | rg -n '^\s*imagePullPolicy:\s' || true)"

echo "deploy: ${NS}/${DEPLOY}"
echo
echo "images:"
echo "${image_lines}"
echo
echo "imagePullPolicy:"
echo "${pull_lines}"
echo

bad=0
if printf '%s\n' "${image_lines}" | rg -q '@sha256:'; then
  echo "ok: deployment uses digest-pinned image reference(s)"
else
  echo "WARN: deployment does not use digest-pinned images (@sha256:...)."
  echo "      If a tag is reused, nodes with IfNotPresent can run stale images."
  bad=1
fi

if printf '%s\n' "${pull_lines}" | rg -qi 'Always'; then
  echo "ok: at least one container uses imagePullPolicy: Always"
else
  echo "WARN: no imagePullPolicy: Always found."
  echo "      If tags are used (not digests), Always is recommended for control-plane services."
  bad=1
fi

exit "${bad}"

