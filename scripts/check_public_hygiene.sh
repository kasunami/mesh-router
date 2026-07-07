#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${ROOT}"

# This script necessarily contains the identifiers it detects, so it excludes
# itself. Generated packaging, cache, coverage, and virtual-environment outputs
# are also outside the public source/test/docs surface.
pattern='(pupix1|Static[-_ ]?Deskix|Static[-_ ]?Mobile[-_ ]?2|Static[-_ ]?Mobilix|mobile[-_ ]?2|tiffs[-_ ]?macbook|packserv|packpup|packhub|packgarage|dawbun|narnia|pi02w|kasunami|/home/kasunami|/Users/kasunami|10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[0-1])\.[0-9]{1,3}\.[0-9]{1,3})'

mapfile -t files < <(
  git ls-files --cached --others --exclude-standard |
    grep -Ev '^(scripts/check_public_hygiene\.sh|mesh_router\.egg-info/|build/|dist/|htmlcov/|\.pytest_cache/|\.venv/|venv/)' |
    grep -Ev '(^|/)(__pycache__/|[^/]+\.pyc$)' |
    while IFS= read -r path; do
      [[ -f "${path}" ]] && printf '%s\n' "${path}"
    done
)

if ((${#files[@]} > 0)); then
  failed=0
  if printf '%s\n' "${files[@]}" | grep -niE -- "${pattern}"; then
    echo "public_hygiene: found private identifiers in public file paths" >&2
    failed=1
  fi
  # The canonical public repository URL is valid package metadata. Replace
  # that exact prefix before scanning so other occurrences of the owner name
  # remain prohibited.
  for path in "${files[@]}"; do
    if sed 's#https://github.com/kasunami/mesh-router#https://github.com/OWNER/mesh-router#g' "${path}" |
      grep -niE -- "${pattern}" | sed "s#^#${path}:#"; then
      echo "public_hygiene: found private identifiers in public file contents" >&2
      failed=1
    fi
  done
  ((failed == 0)) || exit 1
fi

echo "public_hygiene: OK"
