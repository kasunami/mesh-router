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
