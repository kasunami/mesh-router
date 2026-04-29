from __future__ import annotations

from unittest.mock import patch

from mesh_router.app import _is_public_model_name
from mesh_router.router import LaneChoice, LanePlacementError


def test_public_model_name_filter_rejects_support_files() -> None:
    rejected = [
        "config.json",
        "tokenizer.json",
        "added_tokens.json",
        "chat_template.jinja",
        "ggml-vocab-qwen2.gguf",
        "clip_l.safetensors",
        "model-00001-of-00002.safetensors",
        "02ee80b6196926a5ad790a004d9efd6ab1ba6542",
        "Qwen3.5-9B-Q4_K_M.gguf.lock",
    ]
    assert all(not _is_public_model_name(name) for name in rejected)


def test_public_model_name_filter_allows_runnable_names() -> None:
    accepted = [
        "Qwen3.5-9B-Q4_K_M.gguf",
        "falcon3-10b",
        "gemma-4-26B-A4B-it-Q4_K_M",
        "flux1-schnell-Q4_K_S",
    ]
    assert all(_is_public_model_name(name) for name in accepted)


def test_v1_models_only_lists_placeable_public_models(monkeypatch) -> None:
    from mesh_router import app as app_module

    class _Cursor:
        def execute(self, *_args, **_kwargs):
            return None

        def fetchall(self):
            return [
                {"model_name": "ready-model", "tags": ["chat"]},
                {"model_name": "stale-model", "tags": ["chat"]},
                {"model_name": "tokenizer.json", "tags": []},
            ]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class _Conn:
        def cursor(self):
            return _Cursor()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class _DB:
        def connect(self):
            return _Conn()

    def _pick(*, model: str, **_kwargs):
        if model == "stale-model":
            raise LanePlacementError("no READY lanes available serving requested model")
        return LaneChoice(
            lane_id="lane-1",
            worker_id="worker",
            base_url="http://worker.example:11434",
            lane_type="gpu",
            backend_type="llama",
            resolved_model_name=model,
        )

    monkeypatch.setattr(app_module, "db", _DB())
    with patch.object(app_module, "pick_lane_for_model", side_effect=_pick):
        result = app_module.v1_models()

    assert [item["id"] for item in result["data"]] == ["ready-model"]
