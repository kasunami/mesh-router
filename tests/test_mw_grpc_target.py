from __future__ import annotations

import unittest
from unittest.mock import patch

from mesh_router import app as app_module


class FakeCursor:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = list(rows)
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, sql: str, params: tuple) -> None:
        self.executed.append((sql, params))

    def fetchone(self) -> dict:
        if not self._rows:
            return None
        return self._rows.pop(0)


class MwGrpcTargetTests(unittest.TestCase):
    def test_target_requires_control_plane_flag(self) -> None:
        cur = FakeCursor([
            {
                "lane_id": "lane-1",
                "lane_name": "gpu",
                "base_url": "http://10.0.1.99:21434",
                "proxy_auth_metadata": {},
                "host_name": "Static-Deskix",
            }
        ])
        class EmptyMwCursor:
            def execute(self, sql: str, params: tuple) -> None:  # noqa: ARG002
                return None

            def fetchone(self) -> dict | None:
                return None

            def __enter__(self) -> "EmptyMwCursor":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
                return None

        class EmptyMwConn:
            def cursor(self) -> EmptyMwCursor:
                return EmptyMwCursor()

            def __enter__(self) -> "EmptyMwConn":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
                return None

        class EmptyMwDb:
            def connect(self) -> EmptyMwConn:
                return EmptyMwConn()

        with patch.object(app_module, "mw_state_db", EmptyMwDb()):
            target = app_module._mw_target_for_lane(cur=cur, lane_id="lane-1")  # type: ignore[attr-defined]
        self.assertIsNone(target)

    def test_target_derives_endpoint_and_ids(self) -> None:
        cur = FakeCursor([
            {
                "lane_id": "lane-1",
                "lane_name": "gpu",
                "base_url": "http://10.0.1.99:21434",
                "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix"},
                "host_name": "Static-Deskix",
            }
        ])
        target = app_module._mw_target_for_lane(cur=cur, lane_id="lane-1")  # type: ignore[attr-defined]
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.endpoint, f"10.0.1.99:{app_module.settings.mw_grpc_default_port}")  # type: ignore[attr-defined]
        self.assertEqual(target.host_id, "static-deskix")
        self.assertEqual(target.lane_id, "gpu")

    def test_target_infers_mw_for_legacy_cpu_lane_when_state_exists(self) -> None:
        class FakeMwCursor:
            def execute(self, sql: str, params: tuple) -> None:  # noqa: ARG002
                return None

            def fetchone(self) -> dict:
                return {"exists": 1}

            def __enter__(self) -> "FakeMwCursor":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
                return None

        class FakeMwConn:
            def cursor(self) -> FakeMwCursor:
                return FakeMwCursor()

            def __enter__(self) -> "FakeMwConn":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
                return None

        class FakeMwDb:
            def connect(self) -> FakeMwConn:
                return FakeMwConn()

        cur = FakeCursor([
            {
                "lane_id": "lane-cpu",
                "lane_name": "cpu",
                "lane_type": "cpu",
                "base_url": "http://10.0.1.95:21435",
                "proxy_auth_metadata": {},
                "host_name": "pupix1",
            }
        ])
        original_mw_state_db = app_module.mw_state_db
        try:
            app_module.mw_state_db = FakeMwDb()  # type: ignore[assignment]
            target = app_module._mw_target_for_lane(cur=cur, lane_id="lane-cpu")  # type: ignore[attr-defined]
        finally:
            app_module.mw_state_db = original_mw_state_db  # type: ignore[assignment]
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.host_id, "pupix1")
        self.assertEqual(target.lane_id, "cpu")


if __name__ == "__main__":
    unittest.main()
