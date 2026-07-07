#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python -m grpc_tools.protoc \
  --proto_path=proto \
  --python_out=mesh_router/generated \
  --grpc_python_out=mesh_router/generated \
  proto/meshworker.proto

# grpc_tools emits a top-level import. The generated modules live in a Python
# package, so make that import package-relative without changing generated APIs.
python - <<'PY'
from pathlib import Path

path = Path("mesh_router/generated/meshworker_pb2_grpc.py")
generated = path.read_text()
absolute = "import meshworker_pb2 as meshworker__pb2"
relative = "from . import meshworker_pb2 as meshworker__pb2"
if absolute not in generated:
    raise SystemExit(f"expected generated import not found in {path}")
path.write_text(generated.replace(absolute, relative, 1))
PY

echo "Generated Python gRPC modules from proto/meshworker.proto"
