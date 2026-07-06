from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_kubernetes_template_contains_service_and_deployment() -> None:
    documents = list(yaml.safe_load_all((REPO_ROOT / "k8s/mesh-router.yaml").read_text()))

    resources = [
        (document["apiVersion"], document["kind"], document["metadata"]["name"])
        for document in documents
    ]

    assert resources == [
        ("v1", "Service", "mesh-router"),
        ("apps/v1", "Deployment", "mesh-router"),
    ]
