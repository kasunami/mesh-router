#!/usr/bin/env python3
"""Create an inert OpenCode provider patch from MR's selectable model catalog.

The script never makes network calls and never reads or edits active OpenCode
configuration. Review the generated provider-only JSON before any later merge.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

CONTRACT = "mesh-router-selectable-logical-catalog"
INPUT_SAFETY_TOKENS = 2048
MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
MODEL_FIELDS = {"id", "name", "tags", "context_limit", "output_limit", "availability"}


class CatalogError(ValueError):
    pass


def build_patch(document: Any, *, provider_id: str = "homelab-mr") -> dict[str, Any]:
    if not isinstance(document, dict):
        raise CatalogError("catalog must be a JSON object")
    if document.get("contract") != CONTRACT or document.get("scope") != "selectable_logical_models":
        raise CatalogError("input is not MR's selectable logical catalog contract")
    models = document.get("models")
    if not isinstance(models, list) or not models:
        raise CatalogError("catalog models must be a non-empty list")

    configured: dict[str, dict[str, Any]] = {}
    for item in models:
        if not isinstance(item, dict):
            raise CatalogError("each model must be an object")
        if set(item) - MODEL_FIELDS:
            raise CatalogError("catalog entries may not contain host, lane, URL, or unknown fields")
        model_id = item.get("id")
        name = item.get("name")
        tags = item.get("tags", [])
        context = item.get("context_limit")
        output = item.get("output_limit")
        availability = item.get("availability")
        if availability is not None:
            if (not isinstance(availability, dict) or set(availability) -
                    {"present", "loadable", "loaded", "suspended", "dispatch_ready"} or
                    any(type(v) is not bool for v in availability.values())):
                raise CatalogError("invalid model availability state")
        if not isinstance(model_id, str) or not MODEL_ID.fullmatch(model_id):
            raise CatalogError("model id is missing or invalid")
        if model_id in configured:
            raise CatalogError(f"duplicate model id: {model_id}")
        if not isinstance(name, str) or not name.strip():
            raise CatalogError(f"model {model_id} requires a display name")
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise CatalogError(f"model {model_id} has invalid tags")
        if type(context) is not int or context <= 0:
            raise CatalogError(f"model {model_id} has an invalid context limit")
        if type(output) is not int or output <= 0:
            raise CatalogError(
                f"model {model_id} lacks an authoritative positive output_limit; refusing to emit OpenCode model"
            )
        if output >= context:
            raise CatalogError(
                f"model {model_id} has output_limit >= context_limit; refusing an unusable OpenCode token budget"
            )
        input_limit = context - output - INPUT_SAFETY_TOKENS
        if input_limit <= 0:
            raise CatalogError(f"model {model_id} leaves no input after completion and safety reserve")

        model = {
            "name": name.strip(),
            "limit": {"context": context, "output": output, "input": input_limit},
        }
        configured[model_id] = model

    return {"provider": {provider_id: {"models": configured}}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path, help="saved JSON response from GET /api/models/selectable")
    parser.add_argument("patch", type=Path, help="new provider-only patch path; must not already exist")
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.catalog.read_text())
        patch = build_patch(document)
        args.patch.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(args.patch, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(patch, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (OSError, json.JSONDecodeError, CatalogError) as exc:
        print(f"catalog patch refused: {exc}", file=sys.stderr)
        return 2
    print(f"wrote provider-only patch for {len(patch['provider']['homelab-mr']['models'])} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
