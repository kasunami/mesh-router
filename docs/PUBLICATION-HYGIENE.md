# Publication Hygiene

MR should be safe to publish without exposing private topology, local paths, or plaintext secrets.

Tracked files should use placeholders such as:

- `worker-a.example`
- `mesh-router.example`
- `registry.example/mesh-router`
- `/srv/mesh`

Do not commit:

- real hostnames or private device names
- private RFC1918 addresses
- user home paths
- plaintext runtime secrets
- local deployment overlays

Run:

```bash
scripts/check_public_hygiene.sh
```

Private deployment values should live in untracked environment files, sealed Kubernetes secrets, or a private deployment repository.
