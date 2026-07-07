# Contributing

MeshRouter is a working personal-lab project and public reference repository.
Changes should preserve deterministic routing, fail-closed exact placement,
and the separation between routing decisions and model output.

## Development setup

Python 3.11 or newer is required. The locked `uv` workflow is preferred:

```bash
uv sync --python 3.11 --locked --dev
```

For an editable pip environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . pytest build grpcio-tools==1.80.0
```

Do not put deployment credentials in `.env.example`. Local `.env` files are
ignored and process environment variables take precedence.

## Validation

Run the tests and public hygiene gate before submitting a change:

```bash
uv run --no-sync python -m pytest
scripts/check_public_hygiene.sh
```

The pip equivalent for tests is `python -m pytest` after editable installation.

## Protobuf generation

The MeshWorker service contract is tracked in `proto/meshworker.proto`. Generate
the packaged Python modules with the pinned development toolchain:

```bash
uv run --no-sync scripts/generate_proto.sh
git diff --exit-code -- mesh_router/generated
```

Commit the source protocol and generated modules together whenever the contract
changes. Do not edit generated files directly.

## Package and container builds

```bash
uv run --no-sync python -m build
docker build --tag mesh-router:local .
```

The repository CI repeats protobuf generation, tests, hygiene validation,
wheel installation, and the container build on pull requests and pushes to
`main`.

## Change scope

Prefer focused changes with tests. Do not expose private hostnames, addresses,
filesystem paths, credentials, or live topology. Shared-cluster and migration
operations remain deployment concerns and require the operator's normal review
and coordination process.
