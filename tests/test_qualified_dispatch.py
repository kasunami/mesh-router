from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mesh_router import qualified_dispatch as d
from test_qualified_catalog import row


@pytest.fixture
def harness(monkeypatch):
    r = row(loaded=True, lane_id="lane-a", host_id="host-a", host_name="worker-a",
            model_id="model-a", current_model_name="old-model")
    calls = []
    class Reservation:
        def __init__(self, candidate):
            self.row = candidate
            self.stop = SimpleNamespace(is_set=lambda: False)
            calls.append("reserve")
        def start_lease(self, request_id): calls.append("lease")
        def close(self, ok=False): calls.append(("close", ok))
    class Conn:
        @contextmanager
        def cursor(self): yield object()
    class DB:
        @contextmanager
        def connect(self): yield Conn()
    async def collect(**kw):
        calls.append(("inference", kw))
        return {"model": "internal-model", "choices": []}
    def send(**kw):
        calls.append(("load", kw["payload"]["model_name"]))
        return {"ok": True}
    api = SimpleNamespace(_mw_target_for_lane=lambda **_: SimpleNamespace(host_id="worker-a",lane_id="gpu"),
                          _send_mw_command_require_ready=send, _collect_mw_chat_completion=collect)
    monkeypatch.setattr(d, "HostReservation", Reservation)
    monkeypatch.setattr(d, "db", DB())
    monkeypatch.setattr(d, "_current", lambda candidate: candidate)
    monkeypatch.setattr(d, "verify_runtime", lambda candidate: 65536)
    monkeypatch.setattr(d, "prompt_tokens", lambda candidate, payload: 100)
    monkeypatch.setattr(d, "runtime_model", lambda candidate: candidate["model_name"] if candidate["loaded"] else "old-model")
    old = {**r, "model_name": "old-model"}
    monkeypatch.setattr(d.qc, "read_profiles", lambda: [r, old])
    return r, calls, api


def test_ready_instance_is_reused_without_load_and_total_budget_is_preserved(harness):
    r, calls, api = harness
    result = d.dispatch([r], {"messages": [], "max_tokens":4096}, api=api)
    assert result["model"] == r["policy"]["public_id"]
    assert not any(isinstance(c, tuple) and c[0]=="load" for c in calls)
    assert calls.index("reserve") < calls.index("lease")
    infer = next(c[1] for c in calls if isinstance(c,tuple) and c[0]=="inference")
    assert infer["request_payload"]["max_tokens"] == 4096
    assert calls[-1] == ("close", True)


def test_unloaded_profile_reserves_and_leases_before_loading(harness):
    r, calls, api = harness; r["loaded"] = False
    d.dispatch([r], {"messages": [], "max_tokens":100}, api=api)
    assert calls.index("lease") < calls.index(("load", "internal-model"))


def test_load_failure_restores_previous_model_and_releases_reservation(harness):
    r, calls, api = harness; r["loaded"] = False
    def fail_once(**kw):
        model = kw["payload"]["model_name"]; calls.append(("load",model))
        if model == "internal-model": raise RuntimeError("load failed")
        return {"ok": True}
    api._send_mw_command_require_ready = fail_once
    with pytest.raises(HTTPException) as exc:
        d.dispatch([r], {"messages": [], "max_tokens":100}, api=api)
    assert exc.value.status_code == 503
    assert ("load", "old-model") in calls
    assert calls[-1] == ("close", False)


def test_context_revalidated_after_load_before_inference(harness,monkeypatch):
    r,calls,api=harness
    monkeypatch.setattr(d,"prompt_tokens",lambda *_: 16000)
    with pytest.raises(HTTPException) as exc:
        d.dispatch([r], {"messages":[],"max_tokens":4096},api=api)
    assert exc.value.status_code == 422
    assert not any(isinstance(c,tuple) and c[0]=="inference" for c in calls)


def test_busy_or_suspended_host_never_loads(harness):
    r,calls,api=harness;r["loadable"]=False
    with pytest.raises(HTTPException) as exc:
        d.dispatch([r],{"messages":[]},api=api)
    assert exc.value.status_code==409
    assert not calls


def test_larger_request_selects_host_that_honors_advertised_context(harness,monkeypatch):
    r,calls,api=harness
    from copy import deepcopy
    larger=deepcopy(r);larger["host_name"]="worker-b";larger["loaded"]=False
    larger["policy"]["context_limit"]=32768
    selected=[]
    monkeypatch.setattr(d,"_current",lambda c:(selected.append(c["host_name"]) or c))
    monkeypatch.setattr(d,"prompt_tokens",lambda *_:18000)
    d.dispatch([r,larger],{"messages":[{"content":"x"*18000}],"max_tokens":4096},api=api)
    assert selected[0]=="worker-b"


def test_host_pins_rejected_before_any_reservation(harness):
    r,calls,api=harness
    with pytest.raises(HTTPException) as exc:
        d.dispatch([r],{"mesh_pin_worker":"worker-a"},api=api)
    assert exc.value.status_code==400
    assert not calls


def test_stream_disconnect_releases_host_fence(harness):
    import asyncio,json
    r,calls,api=harness
    async def stream(**kwargs):
        yield SimpleNamespace(event_type="delta",raw_backend_payload=json.dumps(
            {"choices":[{"delta":{"content":"OK"}}]}).encode())
        yield SimpleNamespace(event_type="completed",raw_backend_payload=b"[DONE]")
    api.MwGrpcClient=lambda:SimpleNamespace(stream_chat=stream)
    response=d.dispatch([r],{"messages":[],"max_tokens":64,"stream":True},api=api)
    async def consume():
        item=await response.body_iterator.__anext__()
        assert b"OK" in item
        await response.body_iterator.aclose()
    asyncio.run(consume())
    assert calls[-1]==("close",False)


def test_stale_loaded_label_reuses_actual_runtime_without_reloading(harness,monkeypatch):
    r,calls,api=harness;r["loaded"]=False
    monkeypatch.setattr(d,"runtime_model",lambda _:r["model_name"])
    d.dispatch([r],{"messages":[],"max_tokens":64},api=api)
    assert not any(isinstance(c,tuple) and c[0]=="load" for c in calls)


def test_unqualified_actual_model_is_never_evicted(harness,monkeypatch):
    r,calls,api=harness
    monkeypatch.setattr(d,"runtime_model",lambda _:"unknown-artifact")
    with pytest.raises(HTTPException) as exc:
        d.dispatch([r],{"messages":[],"max_tokens":64},api=api)
    assert exc.value.status_code==409
    assert not any(isinstance(c,tuple) and c[0]=="load" for c in calls)
