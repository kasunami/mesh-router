from __future__ import annotations

import unittest

from mesh_router import app as app_module


class FakeCursor:
    def __init__(self, row: dict) -> None:
        self._row = row
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, sql: str, params: tuple) -> None:
        self.executed.append((sql, params))

    def fetchone(self) -> dict:
        return self._row


class MwGrpcTargetTests(unittest.TestCase):
    def test_target_requires_control_plane_flag(self) -> None:
        cur = FakeCursor(
            {
                "lane_id": "lane-1",
                "lane_name": "gpu",
                "base_url": "http://10.0.1.99:21434",
                "proxy_auth_metadata": {},
                "host_name": "Static-Deskix",
            }
        )
        target = app_module._mw_target_for_lane(cur=cur, lane_id="lane-1")  # type: ignore[attr-defined]
        self.assertIsNone(target)

    def test_target_derives_endpoint_and_ids(self) -> None:
        cur = FakeCursor(
            {
                "lane_id": "lane-1",
                "lane_name": "gpu",
                "base_url": "http://10.0.1.99:21434",
                "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix"},
                "host_name": "Static-Deskix",
            }
        )
        target = app_module._mw_target_for_lane(cur=cur, lane_id="lane-1")  # type: ignore[attr-defined]
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.endpoint, f"10.0.1.99:{app_module.settings.mw_grpc_default_port}")  # type: ignore[attr-defined]
        self.assertEqual(target.host_id, "static-deskix")
        self.assertEqual(target.lane_id, "gpu")


if __name__ == "__main__":
    unittest.main()

