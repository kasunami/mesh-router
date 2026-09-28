from __future__ import annotations

import pytest

from mesh_router import app as app_module
from mesh_router import router as router_module
from mesh_router.model_names import canonical_model_name


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Qwen3.5-9B-Q4_K_M.gguf", "qwen3.5-9b"),
        ("qwen3.5-9b", "qwen3.5-9b"),
        ("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", "qwen3.6-35b-a3b"),
        ("qwen3.6-35b-a3b", "qwen3.6-35b-a3b"),
        ("gpt-oss-20b-q4km", "gpt-oss-20b"),
        ("FIM-7B.Q4_K_M.gguf", "fim-7b"),
        ("gemma-4-26b-a4b-qat", "gemma-4-26b-a4b"),
        ("/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", "qwen3.6-35b-a3b"),
    ],
)
def test_canonical_model_name_removes_quant_details(source: str, expected: str) -> None:
    assert canonical_model_name(source) == expected


def test_canonical_id_matches_quantized_host_artifacts() -> None:
    assert app_module._model_request_matches_candidate(
        "qwen3.6-35b-a3b",
        "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    )
    assert router_module._model_matches_request(
        "qwen3.6-35b-a3b",
        "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    )


def test_v1_models_compacts_quantized_aliases_across_ready_hosts(monkeypatch) -> None:
    class _Cursor:
        def execute(self, *_args, **_kwargs):
            return None

        def fetchall(self):
            return [
                {
                    "host_name": "worker-a",
                    "status": "ready",
                    "effective_status": "ready",
                    "current_model_name": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                    "viable_models": [
                        {"model_name": "Qwen3.5-9B-Q4_K_M.gguf", "tags": ["chat"]},
                        {"model_name": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", "tags": ["reasoning"]},
                    ],
                },
                {
                    "host_name": "worker-b",
                    "status": "ready",
                    "effective_status": "ready",
                    "current_model_name": "qwen3.5-9b",
                    "viable_models": [
                        {"model_name": "qwen3.5-9b", "tags": ["chat"]},
                        {"model_name": "qwen3.6-35b-a3b", "tags": ["reasoning"]},
                    ],
                },
                {
                    "host_name": "worker-c",
                    "status": "ready",
                    "effective_status": "ready",
                    "current_model_name": "Qwen3.5-9B-Q5_K_M.gguf",
                    "viable_models": [
                        {"model_name": "Qwen3.5-9B-Q5_K_M.gguf", "tags": ["chat"]},
                        {"model_name": "qwen3.6-35b-a3b", "tags": ["reasoning"]},
                    ],
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
    result = app_module.v1_models()

    assert [entry["id"] for entry in result["data"]] == [
        "qwen3.5-9b",
        "qwen3.6-35b-a3b",
    ]
    assert result["data"][1]["tags"] == ["reasoning"]
