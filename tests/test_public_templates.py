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



def test_kubernetes_template_separates_migration_and_runtime_credentials() -> None:
    documents = list(yaml.safe_load_all((REPO_ROOT / "k8s/mesh-router.yaml").read_text()))
    deployment = next(document for document in documents if document["kind"] == "Deployment")
    pod_spec = deployment["spec"]["template"]["spec"]
    init = next(item for item in pod_spec["initContainers"] if item["name"] == "migrate-db")
    runtime = next(item for item in pod_spec["containers"] if item["name"] == "mesh-router")

    init_env = {item["name"]: item for item in init["env"]}
    assert init_env["MESH_ROUTER_DATABASE_URL"]["valueFrom"]["secretKeyRef"] == {
        "name": "mesh-router-migration-secret",
        "key": "MESH_ROUTER_DATABASE_URL",
    }
    assert init_env["MESH_ROUTER_MW_STATE_DATABASE_URL"]["valueFrom"]["secretKeyRef"] == {
        "name": "mesh-router-migration-secret",
        "key": "MESH_ROUTER_MW_STATE_DATABASE_URL",
    }

    runtime_env = {item["name"]: item for item in runtime["env"]}
    assert runtime_env["MESH_ROUTER_AUTO_MIGRATE_ON_STARTUP"]["value"] == "false"

    runtime_secret_names = {
        source["secretRef"]["name"]
        for source in runtime.get("envFrom", [])
        if "secretRef" in source
    }
    assert "mesh-router-migration-secret" not in runtime_secret_names
