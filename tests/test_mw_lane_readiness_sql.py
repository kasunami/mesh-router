from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from mesh_router import app as app_module
from mesh_router import router as router_module


class MWLaneReadinessSqlTests(unittest.TestCase):
    def test_api_lanes_query_joins_mw_tables_and_uses_actual_model(self) -> None:
        captured: dict[str, object] = {}
        original_db = app_module.db

        class _Cur:
            def execute(self, sql, params=None):  # noqa: ANN001
                captured["sql"] = sql
                captured["params"] = params
                return None

            def fetchall(self):  # noqa: ANN001
                return []

            def __enter__(self):  # noqa: ANN001
                return self

            def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
                return False

        class _Conn:
            def cursor(self):  # noqa: ANN001
                return _Cur()

            def __enter__(self):  # noqa: ANN001
                return self

            def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
                return False

        class _Db:
            def connect(self):  # noqa: ANN001
                return _Conn()

        app_module.db = _Db()  # type: ignore[assignment]
        try:
            client = TestClient(app_module.app)
            resp = client.get("/api/lanes")
            self.assertEqual(resp.status_code, 200)
        finally:
            app_module.db = original_db  # type: ignore[assignment]

        sql = str(captured.get("sql") or "")
        self.assertIn("LEFT JOIN mw_hosts", sql)
        self.assertIn("LEFT JOIN mw_lanes", sql)
        self.assertIn("ml.actual_model", sql)
        self.assertIn("mw_host_id", sql)
        self.assertIn("mw_lane_id", sql)

    def test_pick_lane_for_model_sql_considers_mw_ready_and_uses_actual_model(self) -> None:
        captured: list[str] = []
        original_q = router_module.q
        original_db = router_module.db

        def _fake_q(cur, sql, params=()):  # noqa: ANN001
            captured.append(str(sql))
            return []

        router_module.q = _fake_q  # type: ignore[assignment]
        # Prevent accidental real DB connections during the test.
        class _Cur:
            def __enter__(self):  # noqa: ANN001
                return self

            def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
                return False

        class _Conn:
            def cursor(self):  # noqa: ANN001
                return _Cur()

            def __enter__(self):  # noqa: ANN001
                return self

            def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
                return False

        class _Db:
            def connect(self):  # noqa: ANN001
                return _Conn()

        router_module.db = _Db()  # type: ignore[assignment]
        try:
            with self.assertRaises(Exception):
                router_module.pick_lane_for_model(
                    model="qwen3.5-9b",
                    pin_worker="Static-Deskix",
                )
        finally:
            router_module.q = original_q  # type: ignore[assignment]
            router_module.db = original_db  # type: ignore[assignment]

        sql = "\n".join(captured)
        self.assertIn("LEFT JOIN mw_hosts", sql)
        self.assertIn("LEFT JOIN mw_lanes", sql)
        self.assertIn("COALESCE(ml.actual_model, l.current_model_name)", sql)
        self.assertIn("(l.proxy_auth_metadata->>'control_plane')='mw'", sql)


if __name__ == "__main__":
    unittest.main()
