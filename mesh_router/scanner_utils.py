from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .schemas import ArtifactItem


_SUPPORT_FILE_TOKENS = (
    "embedding",
    "embeddings",
    "embed",
    "tokenizer",
    "mmproj",
    "adapter",
    "lora",
    "vae",
    "clip",
    "text-encoder",
    "text_encoder",
)


def _is_probable_runnable_model_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix not in {".gguf", ".safetensors"}:
        return False
    lowered = path.name.lower()
    if any(token in lowered for token in _SUPPORT_FILE_TOKENS):
        return False
    return True


def scan_model_root(root_path: str) -> List[ArtifactItem]:
    """Scan the local model root for model artifacts."""
    root = Path(root_path)
    artifacts = []

    if not root.exists():
        return artifacts

    # Use os.walk for better control over directory traversal
    for dirpath, dirnames, filenames in os.walk(root):
        path = Path(dirpath)
        
        # Check for MLX model directory signatures
        if "config.json" in filenames and any(
            f.endswith((".safetensors", ".npz")) for f in filenames
        ):
            # Calculate total size of directory
            total_size = sum((Path(dirpath) / f).stat().st_size for f in filenames)
            # Add other files in subdirectories too
            for sub_dirpath, _, sub_filenames in os.walk(dirpath):
                if sub_dirpath == dirpath:
                    continue
                total_size += (sum((Path(sub_dirpath) / f).stat().st_size for f in sub_filenames))

            artifacts.append(
                ArtifactItem(
                    name=path.name,
                    path=str(path.absolute()),
                    format="mlx",
                    size_bytes=total_size,
                    mtime=path.stat().st_mtime,
                    metadata={
                        "type": "mlx_directory",
                    },
                )
            )
            # Skip traversing into this directory further as it's treated as one artifact
            dirnames[:] = []
            continue

        # If not an MLX directory, check for individual model files
        for filename in filenames:
            file_path = path / filename
            if _is_probable_runnable_model_file(file_path):
                artifacts.append(
                    ArtifactItem(
                        name=filename,
                        path=str(file_path.absolute()),
                        format=file_path.suffix.lstrip("."),
                        size_bytes=file_path.stat().st_size,
                        mtime=file_path.stat().st_mtime,
                        metadata={
                            "extension": file_path.suffix,
                        },
                    )
                )

    return artifacts
