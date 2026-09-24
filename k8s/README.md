# mesh-router k8s manifest notes

This directory contains a *template* Service and Deployment manifest
(`mesh-router.yaml`) for deploying MeshRouter into the `ai-tools` namespace.

Validate the public templates without contacting a cluster:

```bash
python -m pytest tests/test_public_templates.py
```

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

`scripts/autodeploy.sh` is an operator action, not a local evaluator command. It
hard-resets its managed clones to their configured remote branches, pushes an
image, commits the resolved digest to a separate manifests repository, and
pushes that repository. Run it only on a dedicated deployment host with the
repository URLs, registry, credentials, and paths explicitly configured.

The systemd example expects the deploy script to be installed at:

- `/usr/local/bin/mesh-router-autodeploy.sh`

For example, install `scripts/autodeploy.sh` to that path on the deployment host.


## Database credentials and migrations

The long-running router should use a restricted runtime database principal from
`mesh-router-secret`. Schema migrations are intentionally run by the
`migrate-db` init container using a separate owner-capable credential from
`mesh-router-migration-secret`.

The Deployment template explicitly overrides both
`MESH_ROUTER_DATABASE_URL` and `MESH_ROUTER_MW_STATE_DATABASE_URL` in the
init container only. The runtime container never receives the migration secret
and sets `MESH_ROUTER_AUTO_MIGRATE_ON_STARTUP=false`.

Create the runtime secret from `mesh-router-secret.example.yaml` and a
separate migration secret with the same two URL keys, for example:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mesh-router-migration-secret
  namespace: ai-tools
type: Opaque
stringData:
  MESH_ROUTER_DATABASE_URL: postgresql://migration_owner:password@meshbrain-db:5432/mesh_router
  MESH_ROUTER_MW_STATE_DATABASE_URL: postgresql://migration_owner:password@meshbrain-db:5432/mesh_router
```

Use your normal secret-management mechanism (for example SealedSecrets) and
never commit real database credentials. Development environments may use the
same principal for both roles, but production should keep runtime DML and
migration DDL credentials separate.
