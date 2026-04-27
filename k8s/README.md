# mesh-router k8s manifest notes

This directory contains a *template* manifest (`mesh-router.yaml`) for deploying Mesh-Router into the `ai-tools` namespace.

## Deploy hygiene (important)

Avoid stale-image hazards:

- Prefer immutable digest references (`registry/repo@sha256:...`) in `image:`.
- If using tags, use unique tags (git SHA) and **never** reuse tags for different digests.
- Do not use `imagePullPolicy: Never` in production-ish environments.

The template uses `imagePullPolicy: Always` and an `image:` placeholder (`__SET_BY_DEPLOY__`) to encourage deploy scripts to set the image explicitly.

## GitOps

For private deployments, the recommended path is a GitOps update of the manifests repo, using:

- `scripts/autodeploy.sh`

That script builds a unique tag, pushes it, resolves the pushed digest, and updates the manifests repo to use the digest.

The systemd example expects the deploy script to be installed at:

- `/usr/local/bin/mesh-router-autodeploy.sh`

For example, install `scripts/autodeploy.sh` to that path on the deployment host.
