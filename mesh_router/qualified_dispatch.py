"""Serialized dispatch for explicitly qualified artifact profiles."""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from typing import Any

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from psycopg.types.json import Jsonb
from starlette.background import BackgroundTask

from . import qualified_catalog as qc
from .config import settings
from .db import db
from .lease_store import heartbeat_router_lease, release_router_lease


class HostReservation:
    """Session advisory lock persists through load, inference and stream close.

    The lease is committed before loading. A process crash releases the lock;
    lease expiry provides the additional recovery fence. All siblings share the
    same lock key, independently of lane names or the selected backend.
    """
    def __init__(self, row: dict[str, Any]):
        self.context = db.connect()
        self.conn = self.context.__enter__()
        self.closed = False
        self.lease_id = None
        self.stop = threading.Event()
        self.thread = None
        self.row = row
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT pg_try_advisory_lock(hashtextextended(%s,0)) AS locked",
                            ("mr-host:" + str(row["host_id"]),))
                if not cur.fetchone()["locked"]:
                    raise RuntimeError("host busy")
                cur.execute("""SELECT 1 FROM router_leases rl JOIN lanes l USING(lane_id)
                               WHERE l.host_id=%s AND rl.state='active' AND rl.expires_at>now() LIMIT 1""",
                            (str(row["host_id"]),))
                if cur.fetchone():
                    raise RuntimeError("host has an active lease")
                # Respect MC/worker leases too; no inference or load may evict
                # an independently leased sibling on the same physical host.
                cur.execute("""SELECT 1 FROM worker_leases wl JOIN worker_lanes w USING(lane_id)
                    WHERE wl.state='active' AND wl.expires_at>now()
                      AND (lower(w.worker_id::text)=lower(%s)
                           OR lower(w.metadata->>'host_id')=lower(%s)) LIMIT 1""",
                            (row["host_name"], row["host_name"]))
                if cur.fetchone():
                    raise RuntimeError("host has an active worker lease")
            self.conn.commit()
        except BaseException:
            self.close(False)
            raise

    def start_lease(self, request_id: str) -> None:
        row = self.row
        with self.conn.cursor() as cur:
            cur.execute("""INSERT INTO router_leases
                (lane_id,model_id,owner,job_type,state,acquired_at,last_heartbeat_at,expires_at,details)
                VALUES (%s,%s,%s,'qualified_chat','active',now(),now(),
                        now()+(%s*interval '1 second'),%s::jsonb) RETURNING lease_id""",
                        (str(row["lane_id"]), str(row["model_id"]), settings.default_owner,
                         settings.default_lease_ttl_seconds, Jsonb({"request_id": request_id,
                         "public_model": row["policy"]["public_id"], "host_exclusive": True})))
            self.lease_id = str(cur.fetchone()["lease_id"])
        self.conn.commit()

        def beat():
            while not self.stop.wait(max(5, settings.default_lease_heartbeat_interval_seconds)):
                try:
                    heartbeat_router_lease(lease_id=self.lease_id)
                except Exception:
                    # The session lock remains the primary concurrency fence.
                    self.stop.set()
        self.thread = threading.Thread(target=beat, name="qualified-lease-heartbeat", daemon=True)
        self.thread.start()

    def close(self, ok: bool = False) -> None:
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        try:
            if self.lease_id:
                release_router_lease(lease_id=self.lease_id, ok=ok)
        finally:
            # Explicit close is required: Db.connect() is not a connection pool.
            self.conn.close()
            self.context.__exit__(None, None, None)


def verify_runtime(row: dict[str, Any]) -> int:
    """Reachability/model/capacity checks are separate from heartbeat freshness."""
    root = str(row["base_url"]).rstrip("/")
    with httpx.Client(timeout=8) as client:
        response = client.get(root + "/health")
        response.raise_for_status()
        response = client.get(root + "/v1/models")
        response.raise_for_status()
    names = {str(x).lower() for x in [row["model_name"], *row["policy"].get("runtime_aliases", [])]}
    for item in response.json().get("data", []):
        if str(item.get("id", "")).lower() in names:
            context = item.get("max_model_len") or item.get("meta", {}).get("n_ctx")
            if type(context) is int and context >= row["policy"]["engine_context"]:
                return context
    raise RuntimeError("runtime model or effective context does not match qualified profile")


def prompt_tokens(row: dict[str, Any], payload: dict[str, Any]) -> int:
    if row["policy"].get("runtime") == "q27":
        # This q27 build has no template/tokenize endpoint. Bound all input
        # bytes and reserve template/MTP room instead of guessing chars/4.
        return qc.conservative_prompt_bound(payload)
    with httpx.Client(timeout=10) as client:
        root = str(row["base_url"]).rstrip("/")
        template = {k: v for k, v in payload.items() if k in
                    ("messages", "tools", "tool_choice", "chat_template_kwargs", "reasoning_format")}
        r = client.post(root + "/apply-template", json=template)
        r.raise_for_status()
        prompt = r.json().get("prompt")
        if not isinstance(prompt, str):
            raise RuntimeError("runtime did not return formatted prompt")
        r = client.post(root + "/tokenize", json={"content": prompt, "add_special": True})
        r.raise_for_status()
        tokens = r.json().get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError("runtime did not return prompt tokens")
        return len(tokens)


def _current(row: dict[str, Any]) -> dict[str, Any]:
    for fresh in qc.read_profiles():
        if str(fresh["lane_id"]) == str(row["lane_id"]) and fresh["policy"]["public_id"] == row["policy"]["public_id"]:
            if fresh["loadable"]:
                return fresh
            break
    raise RuntimeError("placement lost eligibility or heartbeat freshness")


def dispatch(rows: list[dict[str, Any]], payload: dict[str, Any], *, api: Any) -> Any:
    for key in payload:
        if key.startswith("mesh_pin_") and payload[key]:
            raise HTTPException(400, "qualified models use MR-owned placement; remove host/lane pins")
    candidates = [r for r in rows if r["loadable"]]
    bound = qc.conservative_prompt_bound(payload)
    candidates.sort(key=lambda r: (
        not (r["loaded"] and bound + (payload.get("max_tokens") or r["policy"]["output_limit"])
             + r["policy"]["safety_tokens"] <= r["policy"]["context_limit"]),
        -r["policy"]["context_limit"], not r["loaded"], str(r["host_name"])))
    reservation = None
    row = None
    bounded = None
    for candidate in candidates:
        try:
            proposed = qc.bound_payload(payload, candidate["policy"])
        except ValueError:
            continue
        try:
            reservation = HostReservation(candidate)
        except RuntimeError:
            continue
        row, bounded = candidate, proposed
        break
    if reservation is None or row is None:
        raise HTTPException(409, "no idle eligible host supports the requested model/output budget")
    request_id = str(uuid.uuid4())
    previous = str(row.get("current_model_name") or "")
    switched = False
    try:
        row = _current(row)
        reservation.start_lease(request_id)
        with db.connect() as conn:
            with conn.cursor() as cur:
                target = api._mw_target_for_lane(cur=cur, lane_id=str(row["lane_id"]))
        if target is None:
            raise RuntimeError("qualified host has no supported MW control/inference binding")
        if not row["loaded"]:
            if payload.get("mesh_allow_swap") is False:
                raise HTTPException(409, "requested model is unloaded and swapping was disabled")
            switched = True
            api._send_mw_command_require_ready(host_id=target.host_id, message_type="load_model",
                payload={"lane_id": target.lane_id, "model_name": row["model_name"]},
                timeout_seconds=max(120, settings.swap_proxy_timeout_seconds))
        verify_runtime(row)
        # Re-read the source-event clock after potentially slow model loading.
        _current(row)
        bounded = qc.bound_payload(payload, row["policy"])
        for key in list(bounded):
            if key.startswith("mesh_") or key in ("extra_body",):
                bounded.pop(key)
        bounded["model"] = row["model_name"]
        count = prompt_tokens(row, bounded)
        if not qc.fits(bounded, row["policy"], count):
            raise HTTPException(422, "formatted prompt + total completion + safety reserve exceeds model/host context")
        if reservation.stop.is_set():
            raise RuntimeError("lease heartbeat failed before dispatch")
        public_id = row["policy"]["public_id"]
        if not payload.get("stream"):
            bounded["stream"] = False
            result = asyncio.run(api._collect_mw_chat_completion(target=target, request_id=request_id,
                                model=row["model_name"], request_payload=bounded))
            result["model"] = public_id
            reservation.close(True)
            return result

        async def events():
            ok = False
            try:
                async for event in api.MwGrpcClient().stream_chat(target=target, request_id=request_id,
                    model=row["model_name"], messages=bounded.get("messages", []),
                    temperature=bounded.get("temperature"), max_tokens=bounded["max_tokens"],
                    deadline_unix_ms=int((time.time()+settings.default_lease_ttl_seconds)*1000),
                    request_payload={**bounded, "stream": True}):
                    if reservation.stop.is_set():
                        raise RuntimeError("lease heartbeat failed during dispatch")
                    if str(event.event_type or "") in {"error", "failed"}:
                        raise RuntimeError(str(event.error_message or "worker stream failed"))
                    raw = bytes(event.raw_backend_payload or b"").strip()
                    if raw.startswith(b"data:"):
                        raw = raw[5:].strip()
                    if raw == b"[DONE]":
                        break
                    if raw:
                        item = json.loads(raw)
                        item["model"] = public_id
                        yield b"data: " + json.dumps(item).encode() + b"\n\n"
                ok = True
                yield b"data: [DONE]\n\n"
            except Exception as exc:
                yield b"data: " + json.dumps({"error": {"message": str(exc), "type": "dispatch_error"}}).encode() + b"\n\n"
                yield b"data: [DONE]\n\n"
            finally:
                reservation.close(ok)
        return StreamingResponse(events(), media_type="text/event-stream",
            headers={"X-Mesh-Request-Id": request_id}, background=BackgroundTask(reservation.close))
    except BaseException as exc:
        if switched and previous:
            try:
                api._send_mw_command_require_ready(host_id=target.host_id, message_type="load_model",
                    payload={"lane_id": target.lane_id, "model_name": previous},
                    timeout_seconds=max(120, settings.swap_proxy_timeout_seconds))
            except Exception:
                with db.connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute("UPDATE lanes SET status='suspended',suspension_reason='qualified_load_rollback_failed',updated_at=now() WHERE lane_id=%s",
                                    (str(row["lane_id"]),))
                    conn.commit()
        reservation.close(False)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(503, "qualified dispatch failed; prior model restore attempted") from exc
