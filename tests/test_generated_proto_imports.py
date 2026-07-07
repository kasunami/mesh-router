from __future__ import annotations

import importlib


def test_generated_protobuf_modules_import_from_package() -> None:
    pb2 = importlib.import_module("mesh_router.generated.meshworker_pb2")
    pb2_grpc = importlib.import_module("mesh_router.generated.meshworker_pb2_grpc")

    assert pb2.DESCRIPTOR.package == "meshworker.v1"
    assert pb2.ChatRequest.DESCRIPTOR.full_name == "meshworker.v1.ChatRequest"
    assert hasattr(pb2_grpc, "MeshWorkerInferenceStub")
