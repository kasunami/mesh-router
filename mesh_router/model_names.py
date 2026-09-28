from __future__ import annotations

import re


_MODEL_FILE_EXTENSIONS = (".gguf", ".safetensors", ".bin")
_QUANT_MARKER = re.compile(
    r"^(?:"
    r"(?:iq|q)[234568][a-z0-9]*|"
    r"f16|fp16|fp8|bf16|"
    r"int[248]|"
    r"mxfp4|nvfp4|"
    r"qat|ptq[0-9]+[a-z0-9]*"
    r")$",
    re.IGNORECASE,
)
_QUANT_VARIANT_MARKERS = {"ud", "qat"}


def canonical_model_name(model_name: str | None) -> str:
    """Return a lowercase family/version/size ID without file or quant details.

    Examples:
      Qwen3.5-9B-Q4_K_M.gguf -> qwen3.5-9b
      Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -> qwen3.6-35b-a3b
      gpt-oss-20b-q4km -> gpt-oss-20b
    """
    raw = str(model_name or "").strip()
    if not raw:
        return ""
    stem = raw.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    lowered = stem.lower()
    for extension in _MODEL_FILE_EXTENSIONS:
        if lowered.endswith(extension):
            stem = stem[: -len(extension)]
            break

    normalized = stem.strip().lower().replace("_", "-").replace(":", "-")
    # Some inventories separate the architecture size from quantization with a
    # dot (for example, "fim-7b.Q4_K_M"). Preserve version dots such as 3.5,
    # while treating only a dot directly before a quant marker as a separator.
    normalized = re.sub(
        r"\.(?=(?:iq|q)[234568]|(?:f16|fp16|fp8|bf16|int[248]|mxfp4|nvfp4|qat|ptq\d+))",
        "-",
        normalized,
    )
    parts = [part for part in normalized.split("-") if part]
    for index, part in enumerate(parts):
        if _QUANT_MARKER.fullmatch(part) or part in _QUANT_VARIANT_MARKERS:
            return "-".join(parts[:index])
    return "-".join(parts)
