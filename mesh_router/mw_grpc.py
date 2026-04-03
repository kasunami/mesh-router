from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

import grpc

from .generated import meshworker_pb2, meshworker_pb2_grpc


@dataclass(frozen=True)
class MwGrpcTarget:
    endpoint: str  # host:port
    host_id: str
    lane_id: str


class MwGrpcClientError(RuntimeError):
    pass


class MwGrpcClient:
    async def stream_chat(
        self,
        *,
        target: MwGrpcTarget,
        request_id: str,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        deadline_unix_ms: int | None,
    ) -> AsyncIterator[meshworker_pb2.ChatStreamEvent]:
        issued_at_ms = int(time.time() * 1000)
        deadline_ms = deadline_unix_ms or (issued_at_ms + 30_000)

        meta = meshworker_pb2.RequestMeta(
            request_id=request_id,
            host_id=target.host_id,
            lane_id=target.lane_id,
            profile_id="",
            route_type="chat",
            issued_at_unix_ms=issued_at_ms,
            deadline_unix_ms=deadline_ms,
            tags={},
        )
        req = meshworker_pb2.ChatRequest(
            meta=meta,
            model=model,
            messages=[
                meshworker_pb2.ChatMessage(role=str(item.get("role") or ""), content=str(item.get("content") or ""))
                for item in messages
            ],
            temperature=float(temperature or 0.0),
            max_tokens=int(max_tokens or 0),
            stream=True,
            options={},
        )

        async with grpc.aio.insecure_channel(target.endpoint) as channel:
            stub = meshworker_pb2_grpc.MeshWorkerInferenceStub(channel)
            try:
                async for event in stub.StreamChat(req, metadata=[("x-mesh-client-id", "mr")]):
                    yield event
            except grpc.aio.AioRpcError as exc:
                raise MwGrpcClientError(f"MW gRPC StreamChat failed: {exc.code().name}: {exc.details()}") from exc
