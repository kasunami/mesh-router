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


def test_mw_metadata_backend_does_not_make_image_row_ready_for_chat_gpu():
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
            "metadata": {"current_backend_type": "llama"},
        }
    ]

    apply_mw_effective_status(rows, mw_state_db=_FakeDb(mw_rows), stale_seconds=45)

    assert rows[0]["effective_status"] == "offline"
    assert rows[0]["readiness_reason"] == "backend_mismatch"
    assert rows[0]["backend_type"] == "sd"
    assert rows[0]["current_model_name"] == "flux1-schnell-Q4_K_S"


def test_inferred_gpu_lane_stays_offline_when_explicit_image_lane_owns_same_mw_binding():
    now = datetime.now(tz=timezone.utc)
    rows = [
        {
            "lane_id": "gpu-lane",
            "lane_name": "gpu",
            "lane_type": "gpu",
            "backend_type": "llama",
            "host_name": "pupix1",
            "proxy_auth_metadata": {},
            "status": "suspended",
            "current_model_name": "LFM2.5-350M-Q4_K_M.gguf",
        },
        {
            "lane_id": "image-lane",
            "lane_name": "image-gpu",
            "lane_type": "gpu",
            "backend_type": "sd",
            "host_name": "pupix1",
            "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "pupix1", "mw_lane_id": "gpu"},
            "status": "offline",
            "current_model_name": "flux1-schnell-Q4_K_S",
        },
    ]
    mw_rows = [
        {
            "host_id": "pupix1",
            "lane_id": "gpu",
            "last_heartbeat_at": now,
            "actual_state": "running",
            "health_status": "healthy",
            "actual_model": "flux1-schnell-Q4_K_S",
            "backend_type": "stable-diffusion.cpp",
        }
    ]

    apply_mw_effective_status(rows, mw_state_db=_FakeDb(mw_rows), stale_seconds=45)

    assert rows[0]["effective_status"] == "offline"
    assert rows[0]["readiness_reason"] == "backend_mismatch"
    assert rows[0]["backend_type"] == "llama"
    assert rows[0]["current_model_name"] == "LFM2.5-350M-Q4_K_M.gguf"
    assert rows[1]["effective_status"] == "ready"
    assert rows[1]["readiness_reason"] is None
    assert rows[1]["backend_type"] == "sd"
    assert rows[1]["current_model_name"] == "flux1-schnell-Q4_K_S"


def test_mw_overlay_sets_current_model_max_ctx_from_mw_metadata():
    now = datetime.now(tz=timezone.utc)
    rows = [
        {
            "lane_id": "cpu-lane",
            "lane_name": "cpu",
            "lane_type": "cpu",
            "backend_type": "bitnet",
            "host_name": "Static-Deskix",
            "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix", "mw_lane_id": "cpu"},
            "status": "ready",
            "current_model_name": "falcon3-10b",
            "current_model_max_ctx": 2048,
        }
    ]
    mw_rows = [
        {
            "host_id": "static-deskix",
            "lane_id": "cpu",
            "last_heartbeat_at": now,
            "actual_state": "running",
            "health_status": "healthy",
            "actual_model": "falcon3-10b",
            "backend_type": "bitnet.cpp",
            "metadata": {"actual_model_max_ctx": 32768},
        }
    ]

    apply_mw_effective_status(rows, mw_state_db=_FakeDb(mw_rows), stale_seconds=45)

    assert rows[0]["effective_status"] == "ready"
    assert rows[0]["current_model_name"] == "falcon3-10b"
    assert rows[0]["current_model_max_ctx"] == 32768


def test_mw_overlay_overrides_active_backend_and_eta_fields_from_mw_metadata():
    now = datetime.now(tz=timezone.utc)
    rows = [
        {
            "lane_id": "cpu-lane",
            "lane_name": "cpu",
            "lane_type": "cpu",
            "backend_type": "llama",
            "host_name": "Static-Deskix",
            "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix", "mw_lane_id": "cpu"},
            "status": "ready",
            "current_model_name": "LFM2.5-350M-Q4_K_M.gguf",
        }
    ]
    mw_rows = [
        {
            "host_id": "static-deskix",
            "lane_id": "cpu",
            "last_heartbeat_at": now,
            "actual_state": "running",
            "health_status": "healthy",
            "desired_model": "falcon3-10b",
            "actual_model": "falcon3-10b",
            "backend_type": "bitnet.cpp",
            "metadata": {
                "current_backend_type": "bitnet",
                "desired_backend_type": "bitnet",
                "backend_swap_eta_ms": 0,
                "model_swap_eta_ms": 0,
                "total_swap_eta_ms": 0,
                "eta_source": "mw_state_snapshot",
                "eta_complete": True,
            },
        }
    ]

    apply_mw_effective_status(rows, mw_state_db=_FakeDb(mw_rows), stale_seconds=45)

    assert rows[0]["backend_type"] == "bitnet"
    assert rows[0]["current_model_name"] == "falcon3-10b"
    assert rows[0]["desired_model_name"] == "falcon3-10b"
    assert rows[0]["backend_swap_eta_ms"] == 0
    assert rows[0]["model_swap_eta_ms"] == 0
    assert rows[0]["total_swap_eta_ms"] == 0
    assert rows[0]["eta_source"] == "mw_state_snapshot"
    assert rows[0]["eta_complete"] is True


def test_mw_overlay_rewrites_base_url_port_from_live_mw_service_port():
    now = datetime.now(tz=timezone.utc)
    rows = [
        {
            "lane_id": "combined-row",
            "lane_name": "combined",
            "lane_type": "other",
            "backend_type": "llama",
            "host_name": "pupix1",
            "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "pupix1", "mw_lane_id": "combined"},
            "status": "suspended",
            "base_url": "http://10.0.0.95:11436",
            "current_model_name": "gemma-4-26B-A4B-it-Q4_K_M",
        }
    ]
    mw_rows = [
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

    apply_mw_effective_status(rows, mw_state_db=_FakeDb(mw_rows), stale_seconds=45)

    assert rows[0]["effective_status"] == "ready"
    assert rows[0]["current_model_name"] == "Qwen3.5-27B-Q4_K_M"
    assert rows[0]["base_url"] == "http://10.0.0.95:21436"


class _FakeRuntimeStore:
    def __init__(self, facts):
        self._facts = facts

    def get_lane_facts(self, pairs, *, stale_seconds=None):  # noqa: ANN001
        return {pair: self._facts[pair] for pair in pairs if pair in self._facts}


def test_mw_overlay_prefers_runtime_cache_over_stale_db_rows():
    now = datetime.now(tz=timezone.utc)
    rows = [
        {
            "lane_id": "gpu-lane",
            "lane_name": "gpu",
            "lane_type": "gpu",
            "backend_type": "llama",
            "host_name": "Static-Deskix",
            "proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "static-deskix", "mw_lane_id": "gpu"},
            "status": "ready",
            "base_url": "http://10.0.0.99:11434",
            "current_model_name": "stale-db-model",
        }
    ]
    stale_db_rows = [
        {
            "host_id": "static-deskix",
            "lane_id": "gpu",
            "last_heartbeat_at": now,
            "actual_state": "running",
            "health_status": "healthy",
            "actual_model": "stale-db-model",
            "backend_type": "llama.cpp",
            "listen_port": 11434,
        }
    ]
    runtime_store = _FakeRuntimeStore(
        {
            ("static-deskix", "gpu"): {
                "host_id": "static-deskix",
                "lane_id": "gpu",
                "last_heartbeat_at": now,
                "actual_state": "running",
                "health_status": "healthy",
                "actual_model": "qwen3.5-9b",
                "backend_type": "llama.cpp",
                "metadata": {"current_backend_type": "llama"},
                "listen_port": 21434,
            }
        }
    )

    apply_mw_effective_status(rows, mw_state_db=_FakeDb(stale_db_rows), stale_seconds=45, runtime_store=runtime_store)

    assert rows[0]["effective_status"] == "ready"
    assert rows[0]["current_model_name"] == "qwen3.5-9b"
    assert rows[0]["base_url"] == "http://10.0.0.99:21434"
