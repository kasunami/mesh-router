from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# OpenAI-ish request/response models (minimal subset we need)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, alias="max_tokens")
    stream: bool | None = None

    # Router hints (non-OpenAI; accepted via extra body)
    mesh_pin_worker: str | None = None
    mesh_pin_base_url: str | None = None
    mesh_pin_lane_type: str | None = None
    mesh_allow_swap: bool | None = None

    extra_body: dict[str, Any] | None = None

    model_config = {"extra": "allow", "populate_by_name": True}


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "mesh-router"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]

