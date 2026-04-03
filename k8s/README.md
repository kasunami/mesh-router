# mesh-router k8s manifest notes

This directory contains a *template* manifest (`mesh-router.yaml`) for deploying Mesh-Router into the `ai-tools` namespace.

## Deploy hygiene (important)

Avoid stale-image hazards:

- Prefer immutable digest references (`registry/repo@sha256:...`) in `image:`.
- If using tags, use unique tags (git SHA) and **never** reuse tags for different digests.
- Do not use `imagePullPolicy: Never` in production-ish environments.

The template uses `imagePullPolicy: IfNotPresent` and an `image:` placeholder (`__SET_BY_DEPLOY__`) to encourage deploy scripts to set the image explicitly.

## Packserv GitOps

For the homelab setup, the recommended path is a GitOps update of the k3s manifests repo, using:

- `Projects/mesh-router/scripts/packserv_autodeploy.sh`

That script builds a unique tag, pushes it, resolves the pushed digest, and updates the manifests repo to use the digest.

