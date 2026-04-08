from __future__ import annotations

from datetime import datetime, timezone

from mesh_router.mw_overlay import apply_mw_effective_status


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):  # noqa: ANN001
        return None

    def fetchall(self):  # noqa: ANN001
        return self._rows

    def __enter__(self):  # noqa: ANN001
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):  # noqa: ANN001
        return _FakeCursor(self._rows)

    def __enter__(self):  # noqa: ANN001
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class _FakeDb:
    def __init__(self, rows):
        self._rows = rows

    def connect(self):  # noqa: ANN001
        return _FakeConn(self._rows)


def test_explicit_image_lane_stays_offline_when_underlying_mw_lane_is_in_text_backend():
    now = datetime.now(tz=timezone.utc)
    rows = [
        {
            "lane_id": "image-lane",
            "lane_name": "image-gpu",
            "lane_type": "gpu",
            "backend_type": "sd",
            "host_name": "Static-Deskix",
            "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix", "mw_lane_id": "gpu"},
            "status": "offline",
            "current_model_name": "flux1-schnell-Q4_K_S",
        }
    ]
    mw_rows = [
        {
            "host_id": "static-deskix",
            "lane_id": "gpu",
            "last_heartbeat_at": now,
            "actual_state": "running",
            "health_status": "healthy",
            "actual_model": "qwen3.5-9b",
            "backend_type": "llama.cpp",
        }
    ]

    apply_mw_effective_status(rows, mw_state_db=_FakeDb(mw_rows), stale_seconds=45)

    assert rows[0]["effective_status"] == "offline"
    assert rows[0]["readiness_reason"] == "backend_mismatch"
    assert rows[0]["backend_type"] == "sd"
    assert rows[0]["current_model_name"] == "flux1-schnell-Q4_K_S"
