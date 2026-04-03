#!/usr/bin/env bash
set -euo pipefail

NS="${NS:-ai-tools}"
DEPLOY="${DEPLOY:-mesh-router}"
APP_LABEL="${APP_LABEL:-app=mesh-router}"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "fatal: kubectl not found" >&2
  exit 2
fi

echo "verifying deploy hygiene"
echo "  ns:     ${NS}"
echo "  deploy: ${DEPLOY}"

DEPLOY_YAML="$(kubectl -n "${NS}" get deploy "${DEPLOY}" -o yaml)"

unsafe=0

has_digest_refs="$(python3 - <<'PY'
import os, re, sys
text=os.environ["DEPLOY_YAML"]
images=re.findall(r'^\s*image:\s*(\S+)\s*$', text, flags=re.M)
if not images:
  print("no images found")
  sys.exit(1)
all_digest=all("@sha256:" in img for img in images)
print("images:", *images, sep="\n  - ")
print("all_digest=", all_digest)
sys.exit(0 if all_digest else 3)
PY
)" || true

echo "${has_digest_refs}"
if echo "${has_digest_refs}" | rg -q "all_digest= False"; then
  # If not digest-pinned, require imagePullPolicy Always for all containers.
  policies="$(python3 - <<'PY'
import os, re
text=os.environ["DEPLOY_YAML"]
policies=re.findall(r'^\s*imagePullPolicy:\s*(\S+)\s*$', text, flags=re.M)
print("pull_policies:", policies)
print("all_always=", all(p=="Always" for p in policies) if policies else False)
PY
)"
  echo "${policies}"
  if echo "${policies}" | rg -q "all_always= False"; then
    echo "UNSAFE: not digest-pinned and not imagePullPolicy: Always everywhere" >&2
    unsafe=1
  fi
fi

revision="$(echo "${DEPLOY_YAML}" | rg -n "mesh-router\\.openai\\.com/deploy-revision" -o -N || true)"
if [[ -z "${revision}" ]] || echo "${revision}" | rg -q "__SET_BY_DEPLOY__"; then
  echo "WARN: deploy revision annotation missing or placeholder" >&2
fi

echo "pods/imageIDs:"
kubectl -n "${NS}" get pods -l "${APP_LABEL}" -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{range .status.containerStatuses[*]}{.name}{"="}{.imageID}{"\n"}{end}{end}' || true

if [[ "${unsafe}" == "1" ]]; then
  exit 3
fi
echo "OK: deploy hygiene checks passed"

