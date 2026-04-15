from __future__ import annotations

from datetime import datetime, timezone
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from mesh_router import app as app_module
from mesh_router import inventory as inventory_module
from mesh_router.schemas import LaneCapabilityResponse, LaneModelCandidate


class _FakeCursor:
    def __init__(self, *, fetchall_rows: list[list[dict]], fetchone_rows: list[object] | None = None) -> None:
        self._fetchall_rows = fetchall_rows
        self._fetchone_rows = fetchone_rows or []
        self._fa_idx = 0
        self._fo_idx = 0

    def execute(self, sql, params=None):  # noqa: ANN001
        return None

    def fetchall(self):  # noqa: ANN001
        if self._fa_idx >= len(self._fetchall_rows):
            return []
        rows = self._fetchall_rows[self._fa_idx]
        self._fa_idx += 1
        return rows

    def fetchone(self):  # noqa: ANN001
        if self._fo_idx >= len(self._fetchone_rows):
            return None
        row = self._fetchone_rows[self._fo_idx]
        self._fo_idx += 1
        return row

    def __enter__(self):  # noqa: ANN001
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class _FakeConn:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self):  # noqa: ANN001
        return self._cursor

    def __enter__(self):  # noqa: ANN001
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class _FakeDb:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def connect(self):  # noqa: ANN001
        return _FakeConn(self._cursor)


class InventoryApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_db = app_module.db
        self.original_app_mw_state_db = app_module.mw_state_db
        self.original_inventory_mw_state_db = inventory_module.mw_state_db

    def tearDown(self) -> None:
        app_module.db = self.original_db  # type: ignore[assignment]
        app_module.mw_state_db = self.original_app_mw_state_db  # type: ignore[assignment]
        inventory_module.mw_state_db = self.original_inventory_mw_state_db  # type: ignore[assignment]

    def test_api_inventory_overlays_mw_effective_status(self) -> None:
        base_rows = [
            {
                "lane_id": "lane-1",
                "lane_name": "gpu",
                "lane_type": "gpu",
                "backend_type": "llama",
                "base_url": "http://10.0.0.99:11434",
                "status": "offline",
                "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix", "mw_lane_id": "gpu"},
                "current_model_name": None,
                "host_id": "host-1",
                "host_name": "Static-Deskix",
                "viable_models": [
                    {"model_name": "qwen3.5-9b", "tags": [], "max_ctx": 8192, "locality": "local"},
                    {"model_name": "qwen3.5-27b", "tags": [], "max_ctx": 8192, "locality": "remote"},
                ],
            }
        ]
        now = datetime.now(tz=timezone.utc)
        state_rows = [
            {
                "host_id": "static-deskix",
                "lane_id": "gpu",
                "last_heartbeat_at": now,
                "actual_state": "running",
                "health_status": "healthy",
                "actual_model": "Qwen3.5-9B-Q4_K_M.gguf",
            }
        ]

        app_module.db = _FakeDb(_FakeCursor(fetchall_rows=[base_rows]))  # type: ignore[assignment]
        inventory_module.mw_state_db = _FakeDb(_FakeCursor(fetchall_rows=[state_rows]))  # type: ignore[assignment]

        client = TestClient(app_module.app)
        capability_payload = LaneCapabilityResponse(
            lane_id="lane-1",
            capabilities=["chat", "inference"],
            supported_models=["qwen3.5-9b"],
            local_viable_models=[
                LaneModelCandidate(model_name="qwen3.5-9b", tags=[], locality="local"),
            ],
            remote_viable_models=[],
        )
        with mock.patch.object(app_module, "_build_lane_capability_payload", return_value=({}, capability_payload)):
            resp = client.get("/api/inventory")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(len(body["items"]), 1)
        host = body["items"][0]
        self.assertEqual(host["host_name"], "Static-Deskix")
        self.assertIn("stable", host.get("tags") or [])
        self.assertEqual(len(host["lanes"]), 1)
        lane = host["lanes"][0]
        self.assertEqual(lane["lane_name"], "gpu")
        self.assertEqual(lane["status"], "ready")
        self.assertIsNone(lane["readiness_reason"])
        self.assertEqual(lane["current_model_name"], "Qwen3.5-9B-Q4_K_M.gguf")
        self.assertEqual(lane["capabilities"], ["chat", "inference"])
        self.assertEqual(lane["supported_models"], ["qwen3.5-9b"])
        self.assertEqual(len(lane["local_viable_models"]), 1)
        self.assertEqual(len(lane["remote_viable_models"]), 0)

    def test_api_inventory_filters_backend_incompatible_viable_models(self) -> None:
        base_rows = [
            {
                "lane_id": "lane-cpu",
                "lane_name": "cpu",
                "lane_type": "cpu",
                "backend_type": "bitnet",
                "base_url": "http://10.0.0.99:11435",
                "status": "ready",
                "proxy_auth_metadata": {},
                "current_model_name": "falcon3-10b",
                "host_id": "host-1",
                "host_name": "Static-Deskix",
                "viable_models": [
                    {"model_name": "Falcon3-10B-Instruct-1.58bit", "tags": ["bitnet", "cpu"], "locality": "local"},
                    {"model_name": "Qwen3.5-4B-Q4_K_M.gguf", "tags": [], "locality": "local"},
                    {"model_name": "flux1-schnell-Q4_K_S", "tags": [], "locality": "local"},
                ],
            }
        ]

        app_module.db = _FakeDb(_FakeCursor(fetchall_rows=[base_rows]))  # type: ignore[assignment]
        inventory_module.mw_state_db = _FakeDb(_FakeCursor(fetchall_rows=[[]]))  # type: ignore[assignment]

        client = TestClient(app_module.app)
        capability_payload = LaneCapabilityResponse(
            lane_id="lane-cpu",
            capabilities=["chat", "inference"],
            supported_models=["Falcon3-10B-Instruct-1.58bit"],
            local_viable_models=[
                LaneModelCandidate(
                    model_name="Falcon3-10B-Instruct-1.58bit",
                    tags=["bitnet", "cpu"],
                    locality="local",
                ),
            ],
            remote_viable_models=[],
        )
        with mock.patch.object(app_module, "_build_lane_capability_payload", return_value=({}, capability_payload)):
            resp = client.get("/api/inventory")
        self.assertEqual(resp.status_code, 200)
        lane = resp.json()["items"][0]["lanes"][0]
        self.assertEqual(
            [item["model_name"] for item in lane["local_viable_models"]],
            ["Falcon3-10B-Instruct-1.58bit"],
        )

    def test_api_inventory_uses_capability_payload_instead_of_raw_viability_rows(self) -> None:
        base_rows = [
            {
                "lane_id": "lane-mlx",
                "lane_name": "mlx",
                "lane_type": "mlx",
                "backend_type": "llama",
                "base_url": "http://10.0.0.99:11435",
                "status": "ready",
                "proxy_auth_metadata": {},
                "current_model_name": "Qwen3.5-9B-6bit",
                "host_id": "host-1",
                "host_name": "tiffs-macbook",
                "viable_models": [
                    {"model_name": "tokenizer.json", "tags": [], "locality": "local"},
                    {"model_name": "Qwen3.5-9B-6bit", "tags": [], "locality": "local"},
                ],
            }
        ]

        app_module.db = _FakeDb(_FakeCursor(fetchall_rows=[base_rows]))  # type: ignore[assignment]
        inventory_module.mw_state_db = _FakeDb(_FakeCursor(fetchall_rows=[[]]))  # type: ignore[assignment]

        client = TestClient(app_module.app)
        capability_payload = LaneCapabilityResponse(
            lane_id="lane-mlx",
            capabilities=["chat", "inference"],
            supported_models=["Qwen3.5-9B-6bit"],
            current_model="Qwen3.5-9B-6bit",
            local_viable_models=[
                LaneModelCandidate(
                    model_name="Qwen3.5-9B-6bit",
                    tags=[],
                    locality="local",
                    artifact_path="/Users/kasunami/models/Qwen3.5-9B-6bit",
                ),
            ],
            remote_viable_models=[],
        )
        with mock.patch.object(app_module, "_build_lane_capability_payload", return_value=({}, capability_payload)):
            resp = client.get("/api/inventory")
        self.assertEqual(resp.status_code, 200)
        lane = resp.json()["items"][0]["lanes"][0]
        self.assertEqual(
            [item["model_name"] for item in lane["local_viable_models"]],
            ["Qwen3.5-9B-6bit"],
        )

    def test_mesh_inventory_uses_mw_effective_lane_truth(self) -> None:
        now = datetime.now(tz=timezone.utc)
        lanes = [
            {
                "host_name": "Static-Deskix",
                "host_status": "ready",
                "lane_id": "lane-cpu",
                "lane_name": "cpu",
                "lane_type": "cpu",
                "backend_type": "bitnet",
                "base_url": "http://10.0.0.99:11435",
                "lane_status": "offline",
                "suspension_reason": None,
                "current_model_name": "LFM2.5-350M-Q4_K_M.gguf",
                "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix", "mw_lane_id": "cpu"},
                "last_ok_at": None,
                "last_probe_at": None,
            }
        ]
        policy = [
            {"lane_id": "lane-cpu", "model_name": "LFM2.5-350M-Q4_K_M.gguf", "max_ctx": 8192},
        ]
        state_rows = [
            {
                "host_id": "static-deskix",
                "lane_id": "cpu",
                "last_heartbeat_at": now,
                "actual_state": "running",
                "health_status": "healthy",
                "actual_model": "falcon3-10b",
                "backend_type": "bitnet.cpp",
                "listen_port": 21435,
            }
        ]

        app_module.db = _FakeDb(_FakeCursor(fetchall_rows=[lanes, policy, [], []]))  # type: ignore[assignment]
        app_module.mw_state_db = _FakeDb(_FakeCursor(fetchall_rows=[state_rows]))  # type: ignore[assignment]

        client = TestClient(app_module.app)
        resp = client.get("/mesh/inventory")
        self.assertEqual(resp.status_code, 200)
        lane = resp.json()["lanes"][0]
        self.assertEqual(lane["lane_status"], "ready")
        self.assertEqual(lane["raw_lane_status"], "offline")
        self.assertEqual(lane["effective_status"], "ready")
        self.assertIsNone(lane["readiness_reason"])
        self.assertEqual(lane["current_model"], "falcon3-10b")
        self.assertIn("falcon3-10b", lane["known_models"])

    def test_api_lane_lease_status_uses_mw_effective_lane_truth(self) -> None:
        now = datetime.now(tz=timezone.utc)
        cursor = _FakeCursor(
            fetchall_rows=[[]],
            fetchone_rows=[
                {
                    "lane_id": "lane-combined",
                    "lane_name": "combined",
                    "lane_type": "other",
                    "base_url": "http://10.0.0.95:11436",
                    "status": "suspended",
                    "suspension_reason": "swap:abc:queued",
                    "current_model_name": "gemma-4-26B-A4B-it-Q4_K_M",
                    "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "pupix1", "mw_lane_id": "combined"},
                    "backend_type": "llama",
                }
            ],
        )
        state_rows = [
            {
                "host_id": "pupix1",
                "lane_id": "combined",
                "last_heartbeat_at": now,
                "actual_state": "running",
                "health_status": "healthy",
                "actual_model": "Qwen3.5-27B-Q4_K_M",
                "backend_type": "llama.cpp",
                "listen_port": 21436,
            }
        ]

        app_module.db = _FakeDb(cursor)  # type: ignore[assignment]
        app_module.mw_state_db = _FakeDb(_FakeCursor(fetchall_rows=[state_rows]))  # type: ignore[assignment]

        client = TestClient(app_module.app)
        resp = client.get("/api/lanes/lane-combined/lease-status")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["lane_status"], "ready")
        self.assertEqual(body["current_model"], "Qwen3.5-27B-Q4_K_M")
        self.assertEqual(body["base_url"], "http://10.0.0.95:21436")

    def test_api_lane_lease_status_infers_mw_binding_from_host_name(self) -> None:
        now = datetime.now(tz=timezone.utc)
        cursor = _FakeCursor(
            fetchall_rows=[[]],
            fetchone_rows=[
                {
                    "lane_id": "lane-combined",
                    "lane_name": "combined",
                    "lane_type": "other",
                    "base_url": "http://10.0.0.95:11436",
                    "status": "suspended",
                    "suspension_reason": "swap:abc:queued",
                    "current_model_name": "gemma-4-26B-A4B-it-Q4_K_M",
                    "host_name": "pupix1",
                    "proxy_auth_metadata": {},
                    "backend_type": "llama",
                }
            ],
        )
        state_rows = [
            {
                "host_id": "pupix1",
                "lane_id": "combined",
                "last_heartbeat_at": now,
                "actual_state": "running",
                "health_status": "healthy",
                "actual_model": "Qwen3.5-27B-Q4_K_M",
                "backend_type": "llama.cpp",
                "listen_port": 21436,
            }
        ]

        app_module.db = _FakeDb(cursor)  # type: ignore[assignment]
        app_module.mw_state_db = _FakeDb(_FakeCursor(fetchall_rows=[state_rows]))  # type: ignore[assignment]

        client = TestClient(app_module.app)
        resp = client.get("/api/lanes/lane-combined/lease-status")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["lane_status"], "ready")
        self.assertEqual(body["current_model"], "Qwen3.5-27B-Q4_K_M")
        self.assertEqual(body["base_url"], "http://10.0.0.95:21436")


if __name__ == "__main__":
    unittest.main()
