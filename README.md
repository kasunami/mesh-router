# mesh-router

OpenAI-compatible router for LAN worker lanes.

## Safety

This repo is intended to be safe to push to a private GitHub repository.

- Do not commit plaintext secrets.
- Use environment variables for runtime secrets.
- For Kubernetes, commit only placeholder manifests and sealed artifacts.

## Local configuration

Copy `.env.example` to `.env` and fill in real values outside Git.

## Kubernetes secrets

The homelab cluster uses Bitnami Sealed Secrets. A safe workflow is:

1. Create a local plaintext Secret manifest from `k8s/mesh-router-secret.example.yaml`.
2. Seal it with `kubeseal` against the cluster controller.
3. Commit only the resulting sealed manifest.

Example:

```bash
cp k8s/mesh-router-secret.example.yaml /tmp/mesh-router-secret.yaml
$EDITOR /tmp/mesh-router-secret.yaml
kubeseal \
  --controller-name sealed-secrets-controller \
  --controller-namespace kube-system \
  --format yaml \
  < /tmp/mesh-router-secret.yaml \
  > k8s/mesh-router-secret.sealed.yaml
rm -f /tmp/mesh-router-secret.yaml
```

## Build

```bash
scripts/build_image.sh <tag>
```

## Local Model Inventory

To scan a local model root and output a structured inventory for ingestion:

```bash
pip install . psutil  # If not already installed
mesh-router inventory /path/to/models
```

This will output a JSON payload containing identified models (`.gguf`, `.safetensors`, MLX directories) and basic host facts.

To scan an archive model root:

```bash
mesh-router archive-inventory /path/to/archive archive-id --provider packhub
```
