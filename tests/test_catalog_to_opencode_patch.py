from __future__ import annotations

import json

import pytest

from scripts.catalog_to_opencode_patch import CONTRACT, CatalogError, build_patch, main


def _catalog():
    return {
        "contract": CONTRACT,
        "scope": "selectable_logical_models",
        "models": [
            {
                "id": "qwen3.6-35b-a3b",
                "name": "Qwen3.6-35B-A3B",
                "tags": ["chat", "gpu"],
                "context_limit": 32768,
                "output_limit": 4096,
            },
            {
                "id": "small-model",
                "name": "Small Model",
                "tags": ["chat"],
                "context_limit": 8192,
                "output_limit": 1024,
            },
        ],
    }


def test_patch_uses_exact_logical_ids_and_has_no_host_routing():
    patch = build_patch(_catalog())
    assert patch == {
        "provider": {
            "homelab-mr": {
                "models": {
                    "qwen3.6-35b-a3b": {
                        "name": "Qwen3.6-35B-A3B",
                        "limit": {"context": 32768, "output": 4096, "input": 26624},
                    },
                    "small-model": {
                        "name": "Small Model",
                        "limit": {"context": 8192, "output": 1024, "input": 5120},
                    },
                }
            }
        }
    }
    serialized = str(patch).lower()
    assert "host" not in serialized
    assert "lane" not in serialized
    assert "base_url" not in serialized


def test_patch_rejects_ready_only_or_wrong_contract_shape():
    with pytest.raises(CatalogError):
        build_patch({"data": [{"id": "loaded-only"}]})
    doc = _catalog()
    doc["scope"] = "host_loaded_inventory"
    with pytest.raises(CatalogError):
        build_patch(doc)


def test_patch_rejects_host_specific_entries_and_invalid_ids():
    doc = _catalog()
    doc["models"][0]["host_name"] = "worker-a"
    with pytest.raises(CatalogError):
        build_patch(doc)
    doc = _catalog()
    doc["models"][0]["id"] = "../worker-a"
    with pytest.raises(CatalogError):
        build_patch(doc)


def test_patch_rejects_duplicate_ids_and_invalid_context_limits():
    doc = _catalog()
    doc["models"].append(dict(doc["models"][0]))
    with pytest.raises(CatalogError):
        build_patch(doc)
    doc = _catalog()
    doc["models"][0]["context_limit"] = True
    with pytest.raises(CatalogError):
        build_patch(doc)


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_limit", None),
        ("context_limit", 0),
        ("output_limit", None),
        ("output_limit", True),
        ("output_limit", 0),
    ],
)
def test_patch_fails_closed_when_an_authoritative_limit_is_missing_or_invalid(field, value):
    doc = _catalog()
    doc["models"][0][field] = value
    with pytest.raises(CatalogError):
        build_patch(doc)


def test_patch_rejects_output_limit_that_leaves_no_prompt_context():
    doc = _catalog()
    doc["models"][0]["output_limit"] = doc["models"][0]["context_limit"]
    with pytest.raises(CatalogError, match="output_limit >= context_limit"):
        build_patch(doc)


def test_cli_validation_failure_does_not_create_or_replace_patch(tmp_path, capsys):
    catalog_path = tmp_path / "catalog.json"
    patch_path = tmp_path / "provider-patch.json"
    doc = _catalog()
    del doc["models"][0]["output_limit"]
    catalog_path.write_text(json.dumps(doc))

    assert main([str(catalog_path), str(patch_path)]) == 2
    assert "lacks an authoritative positive output_limit" in capsys.readouterr().err
    assert not patch_path.exists()

    patch_path.write_text("pre-existing bytes\n")
    assert main([str(catalog_path), str(patch_path)]) == 2
    assert patch_path.read_text() == "pre-existing bytes\n"


def test_patch_refuses_empty_catalog_to_prevent_empty_provider_overwrite():
    doc = _catalog()
    doc["models"] = []
    with pytest.raises(CatalogError):
        build_patch(doc)


def test_patch_rejects_missing_input_safety_headroom():
    doc = _catalog()
    doc["models"][0]["context_limit"] = 5000
    with pytest.raises(CatalogError, match="no input"):
        build_patch(doc)
