from __future__ import annotations

from mesh_router.app import _is_public_model_name


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


def test_v1_models_lists_ready_inventory_without_live_placement(monkeypatch) -> None:
    from mesh_router import app as app_module

    class _Cursor:
        def execute(self, *_args, **_kwargs):
            return None

        def fetchall(self):
            return [
                {
                    "status": "ready",
                    "effective_status": "ready",
                    "current_model_name": "loaded-model",
                    "viable_models": [
                        {"model_name": "ready-model", "tags": ["chat"]},
                        {"model_name": "tokenizer.json", "tags": []},
                    ],
                },
                {
                    "status": "ready",
                    "effective_status": "offline",
                    "current_model_name": "stale-model",
                    "viable_models": [{"model_name": "stale-model", "tags": ["chat"]}],
                },
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

    monkeypatch.setattr(app_module, "db", _DB())
    monkeypatch.setattr(
        app_module,
        "pick_lane_for_model",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("placement should not run")),
    )
    result = app_module.v1_models()

    assert [item["id"] for item in result["data"]] == ["loaded-model", "ready-model"]
