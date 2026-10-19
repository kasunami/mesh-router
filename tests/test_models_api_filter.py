from __future__ import annotations

from mesh_router.app import _is_public_model_name
from mesh_router.schemas import LaneModelCandidate, LaneCapabilityResponse


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
        "qwen3.5-9b",
        "qwen3.6-35b-a3b",
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
                        {"model_name": "qwen3.6-35b-a3b", "tags": ["chat"]},
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

    assert [item["id"] for item in result["data"]] == ["loaded-model"]


def test_selectable_catalog_is_logical_local_verified_and_read_only(monkeypatch) -> None:
    from mesh_router import app as app_module

    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class _Conn:
        def __init__(self):
            self.commits = 0

        def cursor(self):
            return _Cursor()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def commit(self):
            self.commits += 1

    conn = _Conn()

    class _DB:
        def connect(self):
            return conn

    monkeypatch.setattr(app_module, "db", _DB())
    monkeypatch.setattr(
        app_module,
        "fetch_lane_inventory",
        lambda *, cur: [{"lane_id": "lane-a"}, {"lane_id": "lane-b"}],
    )
    calls = []

    def build_capability(cur, lane_id, *, persist_viability=True):
        calls.append((lane_id, persist_viability))
        if lane_id == "lane-a":
            candidates = [
                LaneModelCandidate(
                    model_name="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                    tags=["chat", "llama"],
                    locality="local",
                    artifact_path="/models/qwen-a.gguf",
                    artifact_host="worker-a",
                    max_context_tokens=32768,
                ),
                LaneModelCandidate(
                    model_name="loaded-without-artifact",
                    tags=["chat"],
                    locality="local",
                    artifact_path=None,
                    artifact_host="worker-a",
                ),
            ]
        else:
            candidates = [
                LaneModelCandidate(
                    model_name="qwen3.6-35b-a3b-q8_0.gguf",
                    tags=["chat", "gpu"],
                    locality="local",
                    artifact_path="/models/qwen-b.gguf",
                    artifact_host="worker-b",
                    max_context_tokens=65536,
                ),
                LaneModelCandidate(
                    model_name="remote-only-model",
                    tags=["chat"],
                    locality="remote",
                    artifact_path="/archive/remote.gguf",
                    artifact_host="archive",
                ),
                LaneModelCandidate(
                    model_name="unverified-model",
                    tags=["chat"],
                    locality="unverified",
                    artifact_path="/models/unverified.gguf",
                    artifact_host="worker-b",
                ),
            ]
        return {}, LaneCapabilityResponse(
            lane_id=lane_id,
            capabilities=["chat"],
            supported_models=[candidate.model_name for candidate in candidates],
            local_viable_models=[candidate for candidate in candidates if candidate.locality == "local"],
            remote_viable_models=[candidate for candidate in candidates if candidate.locality == "remote"],
            unverified_models=[candidate for candidate in candidates if candidate.locality == "unverified"],
        )

    monkeypatch.setattr(app_module, "_build_lane_capability_payload", build_capability)
    result = app_module.api_selectable_logical_models()

    assert result["contract"] == "mesh-router-selectable-logical-catalog"
    assert result["scope"] == "selectable_logical_models"
    assert result["models"] == [
        {
            "id": "qwen3.6-35b-a3b",
            "name": "qwen3.6-35b-a3b",
            "tags": ["chat", "gpu", "llama"],
            "context_limit": 32768,
        }
    ]
    assert calls == [("lane-a", False), ("lane-b", False)]
    assert conn.commits == 0
    assert not ({"host", "host_id", "host_name", "lane_id", "base_url"} & set(result["models"][0]))
