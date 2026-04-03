from __future__ import annotations

import contextlib
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from mesh_router import app as app_module


class FakeCursor:
    def __init__(self) -> None:
        self._row: dict | None = None

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def execute(self, sql: str, params: tuple | None = None) -> None:  # noqa: ARG002
        # Return a deterministic fake row.
        self._row = {
            "request_id": "11111111-1111-1111-1111-111111111111",
            "host_id": "static-deskix",
            "transition_type": "activate_profile",
            "status": "completed",
            "current_phase": "done",
            "error_kind": None,
            "error_message": None,
            "started_at": datetime(2026, 4, 3, 12, 0, 0, tzinfo=UTC),
            "completed_at": datetime(2026, 4, 3, 12, 0, 1, tzinfo=UTC),
            "updated_at": datetime(2026, 4, 3, 12, 0, 1, tzinfo=UTC),
        }

    def fetchone(self) -> dict | None:
        return self._row


class FakeConn:
    def __enter__(self) -> "FakeConn":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def cursor(self) -> FakeCursor:
        return FakeCursor()


@contextlib.contextmanager
def fake_connect():
    yield FakeConn()


class MwCommandStatusPollDbTests(unittest.TestCase):
    def test_status_endpoint_queries_mw_state_db_not_core_db(self) -> None:
        fake_mw_state_db = SimpleNamespace(connect=fake_connect)

        # If core DB is used here, fail loudly. Polling must be aligned with mw_state_db.
        class ExplodingDb:
            def connect(self):  # noqa: ANN201
                raise AssertionError("core db.connect() must not be called by /api/mw/commands/{request_id}")

        with (
            patch.object(app_module, "db", ExplodingDb()),
            patch.object(app_module, "mw_state_db", fake_mw_state_db),
        ):
            client = TestClient(app_module.app)
            resp = client.get("/api/mw/commands/11111111-1111-1111-1111-111111111111")
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertTrue(body.get("found"))
            self.assertEqual(body.get("status"), "completed")
            self.assertEqual(body.get("ok"), True)


if __name__ == "__main__":
    unittest.main()

