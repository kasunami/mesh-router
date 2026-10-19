from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mesh_router import lease_store as ls, qualified_dispatch as qd
from mesh_router.request_store import router_request_health, serialize_router_request


class Cursor:
    def __init__(self, found=None):
        self.calls = []
        self.found = found
    def execute(self, sql, params=None): self.calls.append((sql, params))
    def fetchone(self): return self.found
    def __enter__(self): return self
    def __exit__(self, *_): pass


class Connection:
    def __init__(self, cur): self.cur = cur; self.commits = 0
    def cursor(self): return self.cur
    def commit(self): self.commits += 1


def fake_db(monkeypatch, module, cur):
    conn = Connection(cur)
    @contextmanager
    def connect(): yield conn
    monkeypatch.setattr(module, "db", SimpleNamespace(connect=connect))
    return conn


def test_qualified_lease_registers_same_request_before_commit(monkeypatch):
    cur = Cursor({"lease_id": "lease-1"}); conn = Connection(cur)
    reservation = object.__new__(qd.HostReservation)
    reservation.conn = conn
    reservation.row = dict(lane_id="lane-1", model_id="model-1", host_name="worker-a",
                           model_name="artifact-alias", base_url="http://backend", policy={"public_id":"logical-model"})
    monkeypatch.setattr(qd.threading, "Thread", Mock())
    reservation.start_lease("request-1")
    assert len(cur.calls) == 2
    sql, params = cur.calls[1]
    assert "INSERT INTO router_requests" in sql
    assert "'running'" in sql
    assert params == ("request-1", "logical-model", "artifact-alias", "worker-a", "http://backend", "lease-1")
    assert conn.commits == 1


def test_lease_heartbeat_refreshes_qualified_request_without_inference_progress(monkeypatch):
    cur = Cursor(); conn = fake_db(monkeypatch, ls, cur)
    ls.heartbeat_router_lease(lease_id="lease-1")
    assert len(cur.calls) == 2
    assert "UPDATE router_requests" in cur.calls[1][0]
    assert "qualified_chat" in cur.calls[1][0]
    assert cur.calls[1][1] == ("lease-1",)
    assert conn.commits == 1


@pytest.mark.parametrize("ok", [True, False])
def test_release_updates_request_terminal_state_and_preserves_cancellation(monkeypatch, ok):
    cur = Cursor(); fake_db(monkeypatch, ls, cur)
    ls.release_router_lease(lease_id="lease-1", ok=ok)
    assert cur.calls[0][1] == ("released" if ok else "failed", "lease-1")
    assert "rr.cancel_requested" in cur.calls[1][0]
    assert "released_at=rl.released_at" in cur.calls[1][0]


@pytest.mark.parametrize("found", [None, {"present":1}])
def test_control_admission_rejects_any_active_host_lease(monkeypatch, found):
    cur = Cursor(found); fake_db(monkeypatch, ls, cur)
    if found:
        with pytest.raises(RuntimeError, match="host busy"):
            ls.require_idle_control_host("worker-a")
    else:
        ls.require_idle_control_host("worker-a")
    assert "JOIN lanes" in cur.calls[0][0] and "JOIN hosts" in cur.calls[0][0]
    assert cur.calls[0][1] == ("worker-a",)


def test_registered_prefill_is_active_and_healthy_to_orphan_detector():
    now = datetime.now(UTC)
    row = dict(request_id="request-1", state="running", lease_state="active",
               acquired_at=now-timedelta(minutes=2), last_heartbeat_at=now,
               expires_at=now+timedelta(minutes=1))
    status = serialize_router_request(row); health = router_request_health(row)
    assert status["state"] == "running" and not status["terminal"]
    assert health["healthy"] and health["lease_active"] and not health["terminal"]
