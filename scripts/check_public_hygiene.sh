#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${ROOT}"

pattern='(pupix1|Static-Deskix|static-deskix|Static-Mobile-2|static-mobile-2|mobile-2|tiffs-macbook|packserv|packpup|packhub|packgarage|dawbun|narnia|pi02w|kasunami|/home/kasunami|/Users/kasunami|10\.0\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[0-1])\.)'

mapfile -t files < <(
  git ls-files --cached --others --exclude-standard |
    grep -Ev '^(tests/|mesh_router\.egg-info/|scripts/check_public_hygiene\.sh$)' |
    grep -Ev '\.pyc$' |
    while IFS= read -r path; do
      [[ -f "${path}" ]] && printf '%s\n' "${path}"
    done
)

if ((${#files[@]} > 0)) && grep -nE -- "${pattern}" "${files[@]}"; then
  echo "public_hygiene: found private identifiers in public files" >&2
  exit 1
fi

echo "public_hygiene: OK"
