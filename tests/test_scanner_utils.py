from __future__ import annotations

from pathlib import Path

from mesh_router.scanner_utils import scan_model_root


def test_scan_model_root_filters_support_files(tmp_path: Path) -> None:
    (tmp_path / "Qwen3.5-4B-Q4_K_M.gguf").write_bytes(b"model")
    (tmp_path / "nomic-embed-text.gguf").write_bytes(b"embedding")
    (tmp_path / "tokenizer.gguf").write_bytes(b"tokenizer")
    (tmp_path / "model-mmproj.gguf").write_bytes(b"projector")
    (tmp_path / "adapter.safetensors").write_bytes(b"adapter")

    names = {item.name for item in scan_model_root(str(tmp_path))}

    assert names == {"Qwen3.5-4B-Q4_K_M.gguf"}


def test_scan_model_root_preserves_mlx_directory(tmp_path: Path) -> None:
    model_dir = tmp_path / "mlx-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "weights.safetensors").write_bytes(b"weights")

    artifacts = scan_model_root(str(tmp_path))

    assert len(artifacts) == 1
    assert artifacts[0].name == "mlx-model"
    assert artifacts[0].format == "mlx"
