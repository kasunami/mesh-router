from __future__ import annotations

from datetime import datetime, timezone
import unittest

from fastapi.testclient import TestClient

from mesh_router import app as app_module


class _Cur:
    def __init__(self) -> None:
        self._fetchone_queue: list[object] = []
        self._fetchall_queue: list[list[dict]] = []
        self.executed: list[tuple[str, object | None]] = []

    def queue_fetchone(self, v: object) -> None:
        self._fetchone_queue.append(v)

    def queue_fetchall(self, rows: list[dict]) -> None:
        self._fetchall_queue.append(rows)

    def execute(self, sql, params=None):  # noqa: ANN001
        self.executed.append((str(sql), params))

    def fetchone(self):  # noqa: ANN001
        return self._fetchone_queue.pop(0) if self._fetchone_queue else None

    def fetchall(self):  # noqa: ANN001
        return self._fetchall_queue.pop(0) if self._fetchall_queue else []

    def __enter__(self):  # noqa: ANN001
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class _Conn:
    def __init__(self, cur: _Cur) -> None:
        self._cur = cur

    def cursor(self):  # noqa: ANN001
        return self._cur

    def __enter__(self):  # noqa: ANN001
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class _Db:
    def __init__(self, cur: _Cur) -> None:
        self._cur = cur

    def connect(self):  # noqa: ANN001
        return _Conn(self._cur)


class PerfRegistryApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_mw_state_db = app_module.mw_state_db

    def tearDown(self) -> None:
        app_module.mw_state_db = self.original_mw_state_db  # type: ignore[assignment]

    def test_ingest_then_expectations_best_effort(self) -> None:
        cur = _Cur()
        # insert_observation: table_exists -> fetchone truthy, then INSERT (no fetch)
        cur.queue_fetchone({"ok": 1})
        # get_expectation: table_exists -> fetchone truthy, then rows
        cur.queue_fetchone({"ok": 1})
        now = datetime.now(tz=timezone.utc)
        cur.queue_fetchall(
            [
                {"observed_at": now, "first_token_ms": 12.0, "decode_tps": 100.0, "total_ms": 40.0},
                {"observed_at": now, "first_token_ms": 14.0, "decode_tps": 90.0, "total_ms": 50.0},
            ]
        )

        app_module.mw_state_db = _Db(cur)  # type: ignore[assignment]
        client = TestClient(app_module.app)

        resp = client.post(
            "/api/perf/observations",
            json={
                "host_id": "static-deskix",
                "lane_id": "lane-1",
                "model_name": "qwen3.5-9b",
                "modality": "chat",
                "first_token_ms": 13.0,
                "decode_tps": 95.0,
                "ok": True,
                "metadata": {"source": "unit-test"},
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json().get("ok"))

        resp2 = client.get(
            "/api/perf/expectations",
            params={"host_id": "static-deskix", "lane_id": "lane-1", "model_name": "qwen3.5-9b", "modality": "chat"},
        )
        self.assertEqual(resp2.status_code, 200)
        items = resp2.json()["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["sample_count"], 2)
        self.assertIsNotNone(items[0].get("decode_tps_p50"))


if __name__ == "__main__":
    unittest.main()
