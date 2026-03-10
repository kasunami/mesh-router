from __future__ import annotations

import time
from typing import Any

from datetime import UTC, datetime, timedelta

import httpx
from fastapi import Body, FastAPI, Header, HTTPException, Request
from psycopg.types.json import Jsonb
from pydantic import BaseModel

from .config import settings
from .db import db
from .router import pick_lane_for_model
from .schemas import ChatCompletionRequest, ModelsResponse, ModelInfo
from .tokens import sign_token, verify_token


app = FastAPI(title="mesh-router", version="0.1.0")

def _strip_nones(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if v is None:
                continue
            vv = _strip_nones(v)
            # Drop empty dicts/lists introduced by stripping.
            if vv == {} or vv == []:
                continue
            out[k] = vv
        return out
    if isinstance(value, list):
        return [_strip_nones(v) for v in value if v is not None]
    return value


def _downstream_payload(req: ChatCompletionRequest) -> dict[str, Any]:
    raw = req.model_dump(by_alias=True)
    # Remove router-only hint fields.
    for k in list(raw.keys()):
        if k.startswith("mesh_") or k in {"extra_body"}:
            raw.pop(k, None)
    return _strip_nones(raw)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True}


@app.get("/health/liveliness")
def health_liveliness() -> dict[str, Any]:
    # Compatibility endpoint (LiteLLM-style health paths are used by some clients).
    return {"ok": True}


@app.get("/health/readiness")
def health_readiness() -> dict[str, Any]:
    return {"ok": True}


@app.get("/v1/models")
def v1_models() -> dict[str, Any]:
    def _is_canonical(model: str) -> bool:
        m = (model or "").strip()
        if not m:
            return False
        # Avoid downstream aliases like filesystem paths or URLs.
        if "/" in m or "\\" in m:
            return False
        if "://" in m:
            return False
        if len(m) > 128:
            return False
        return True

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT model_name FROM models ORDER BY model_name")
            rows = cur.fetchall()
    data = [ModelInfo(id=str(r["model_name"])) for r in rows if _is_canonical(str(r["model_name"]))]
    resp = ModelsResponse(data=data)
    return resp.model_dump()


def _bearer_token(req: Request) -> str:
    auth = (req.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth.split(None, 1)[1].strip()
    return ""


def _cleanup_expired_router_leases(cur) -> None:
    cur.execute(
        "UPDATE router_leases SET state='expired' WHERE state='active' AND expires_at < now()"
    )


def _acquire_router_lease(
    *,
    lane_id: str,
    model_id: str,
    owner: str,
    job_type: str,
    ttl_seconds: int,
    details: dict[str, Any],
) -> tuple[str, datetime]:
    expires_at = datetime.now(UTC) + timedelta(seconds=max(30, int(ttl_seconds)))
    with db.connect() as conn:
        with conn.cursor() as cur:
            _cleanup_expired_router_leases(cur)
            cur.execute(
                """
                SELECT 1
                FROM router_leases
                WHERE lane_id=%s AND state='active' AND expires_at > now()
                LIMIT 1
                """,
                (lane_id,),
            )
            if cur.fetchone():
                raise RuntimeError("lane busy")
            cur.execute(
                """
                INSERT INTO router_leases (lane_id, model_id, owner, job_type, state, expires_at, details)
                VALUES (%s, %s, %s, %s, 'active', %s, %s::jsonb)
                RETURNING lease_id
                """,
                (lane_id, model_id, owner, job_type, expires_at, Jsonb(details)),
            )
            lease_id = str(cur.fetchone()["lease_id"])
        conn.commit()
    return lease_id, expires_at


def _release_router_lease(*, lease_id: str, ok: bool) -> None:
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE router_leases
                SET state=%s, released_at=now()
                WHERE lease_id=%s AND state='active'
                """,
                ("released" if ok else "failed", lease_id),
            )
        conn.commit()


@app.post("/api/router-leases/validate")
def api_router_lease_validate(
    req: Request,
    body: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    """Validate a router-issued lease token. Intended for worker-token-gateway.py."""
    token = _bearer_token(req)
    if not token:
        raise HTTPException(status_code=401, detail="missing bearer token")
    expected_model = None
    try:
        expected_model = str((body or {}).get("model") or "").strip() or None
    except Exception:
        expected_model = None
    try:
        claims = verify_token(token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

    lease_id = str(claims.get("lease_id") or "").strip()
    lane_id = str(claims.get("lane_id") or "").strip()
    claim_model = str(claims.get("model") or "").strip() or None
    if not lease_id or not lane_id:
        raise HTTPException(status_code=401, detail="invalid token claims")
    if expected_model and claim_model and expected_model != claim_model:
        raise HTTPException(status_code=401, detail="model mismatch")

    with db.connect() as conn:
        with conn.cursor() as cur:
            _cleanup_expired_router_leases(cur)
            cur.execute(
                """
                SELECT lease_id, lane_id, state, expires_at
                FROM router_leases
                WHERE lease_id=%s AND lane_id=%s
                """,
                (lease_id, lane_id),
            )
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="lease not found")
    if str(row["state"]) != "active":
        raise HTTPException(status_code=401, detail="lease not active")
    return {"ok": True, **claims}


@app.post("/v1/chat/completions")
def v1_chat_completions(
    req: ChatCompletionRequest,
    x_mesh_pin_worker: str | None = Header(default=None),
    x_mesh_pin_base_url: str | None = Header(default=None),
) -> dict[str, Any]:
    # Pins: header overrides body.
    pin_worker = x_mesh_pin_worker or req.mesh_pin_worker
    pin_base_url = x_mesh_pin_base_url or req.mesh_pin_base_url
    pin_lane_type = req.mesh_pin_lane_type

    try:
        choice = pick_lane_for_model(
            model=req.model,
            pin_worker=pin_worker,
            pin_base_url=pin_base_url,
            pin_lane_type=pin_lane_type,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    lease = None
    started = time.time()

    # Ensure model exists in our DB for metrics and resolve any per-lane downstream alias.
    downstream_model = req.model
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO models (model_name, format)
                VALUES (%s, 'other'::model_format)
                ON CONFLICT (model_name) DO UPDATE SET updated_at=now()
                """,
                (req.model,),
            )
            cur.execute("SELECT model_id FROM models WHERE model_name=%s", (req.model,))
            model_id = cur.fetchone()["model_id"]
            lane_id = choice.lane_id or None

            if lane_id is not None:
                cur.execute(
                    """
                    SELECT downstream_model_name
                    FROM lane_model_aliases
                    WHERE lane_id=%s AND model_id=%s
                    """,
                    (lane_id, model_id),
                )
                alias_row = cur.fetchone()
                if alias_row and alias_row.get("downstream_model_name"):
                    downstream_model = str(alias_row["downstream_model_name"])
        conn.commit()

    ok = True
    err_kind = None
    err_msg = None
    resp_data: dict[str, Any] | None = None
    tps: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    try:
        # Acquire a router lease (the only source of truth). The worker gateway will
        # call back to /api/router-leases/validate with this token.
        lease_id, expires_at = _acquire_router_lease(
            lane_id=choice.lane_id,
            model_id=str(model_id),
            owner=settings.default_owner,
            job_type=settings.default_job_type,
            ttl_seconds=settings.default_lease_ttl_seconds,
            details={
                "worker_id": choice.worker_id,
                "base_url": choice.base_url,
                "downstream_model": downstream_model,
            },
        )
        lease = {
            "lease_id": lease_id,
            "expires_at": expires_at.isoformat(),
        }
        token = sign_token(
            {
                "lease_id": lease_id,
                "lane_id": choice.lane_id,
                "worker_id": choice.worker_id,
                "base_url": choice.base_url,
                "model": downstream_model,
                "owner": settings.default_owner,
                "exp": int(expires_at.timestamp()),
            }
        )
        payload = _downstream_payload(req)
        # Apply downstream model alias if needed (must match leased model for worker gateway validation).
        payload["model"] = downstream_model
        # Proxy directly to the worker gateway (11434/11435). Backends are bound to localhost.
        with httpx.Client(timeout=600.0) as client:
            r = client.post(
                f"{choice.base_url.rstrip('/')}/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
            )
            try:
                resp_data = r.json()
            except Exception:
                resp_data = {"raw": r.text}
            if r.status_code >= 400:
                raise RuntimeError(f"worker proxy http_{r.status_code}: {resp_data}")

        # Best-effort parse of llama.cpp timings/usage when available.
        try:
            usage = (resp_data or {}).get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens")) if usage.get("prompt_tokens") is not None else None
            completion_tokens = int(usage.get("completion_tokens")) if usage.get("completion_tokens") is not None else None
        except Exception:
            pass
        try:
            timings = (((resp_data or {}).get("timings") or {}) if isinstance(resp_data, dict) else {})
            if timings.get("predicted_per_second") is not None:
                tps = float(timings["predicted_per_second"])
        except Exception:
            pass

        return resp_data
    except Exception as e:
        ok = False
        err_kind = "proxy_error"
        err_msg = str(e)
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        elapsed_ms = int((time.time() - started) * 1000)
        # Metrics: best-effort insert.
        try:
            with db.connect() as conn:
                with conn.cursor() as cur:
                    if lane_id is not None:
                        cur.execute(
                            """
                            INSERT INTO lane_model_metrics (
                              lane_id, model_id,
                              load_time_ms, request_latency_ms,
                              tps, prompt_tokens, completion_tokens,
                              success, error_kind, error_message, run_tag
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                lane_id,
                                model_id,
                                None,
                                elapsed_ms,
                                tps,
                                prompt_tokens,
                                completion_tokens,
                                ok,
                                err_kind,
                                err_msg,
                                "mesh-router:chat",
                            ),
                        )
                conn.commit()
        except Exception:
            pass
        try:
            if lease is not None:
                _release_router_lease(lease_id=str(lease["lease_id"]), ok=ok)
        except Exception:
            pass


class SwapModelRequest(BaseModel):
    model_name: str


@app.post("/api/lanes/{lane_id}/swap-model")
def api_lane_swap_model(lane_id: str, req: SwapModelRequest) -> dict[str, Any]:
    """
    Ask the worker gateway for a specific lane to swap its loaded model.
    The gateway updates the mesh-llama-launch config and restarts the systemd service.
    On success, updates current_model_name in the DB.
    """
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT base_url FROM lanes WHERE lane_id=%s", (lane_id,))
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="lane not found")

    base_url = str(row["base_url"])
    try:
        # Allow generous time: swap involves systemctl restart + model load from disk.
        with httpx.Client(timeout=180.0) as client:
            r = client.post(
                f"{base_url.rstrip('/')}/swap-model",
                json={"model_name": req.model_name},
                headers={"Authorization": f"Bearer {settings.swap_auth_token}"},
            )
            try:
                data = r.json()
            except Exception:
                data = {"raw": r.text}
            if r.status_code >= 400:
                raise HTTPException(status_code=r.status_code, detail=str(data))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE lanes SET current_model_name=%s, updated_at=now() WHERE lane_id=%s",
                (req.model_name, lane_id),
            )
        conn.commit()

    return {"ok": True, "lane_id": lane_id, "model_name": req.model_name}


@app.get("/mesh/inventory")
def mesh_inventory() -> dict[str, Any]:
    """Best-effort inventory view: hosts, lanes, and known per-lane model options."""
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  h.host_name,
                  h.status as host_status,
                  l.lane_id,
                  l.lane_type,
                  l.base_url,
                  l.status as lane_status,
                  l.current_model_name,
                  l.last_ok_at,
                  l.last_probe_at
                FROM lanes l
                JOIN hosts h ON h.host_id=l.host_id
                ORDER BY h.host_name, l.lane_type, l.base_url
                """
            )
            lanes = cur.fetchall()
            # Allowed/known models per lane (policy + aliases).
            cur.execute(
                """
                SELECT l.lane_id, m.model_name
                FROM lane_model_policy p
                JOIN lanes l ON l.lane_id=p.lane_id
                JOIN models m ON m.model_id=p.model_id
                WHERE p.allowed=true
                """
            )
            policy = cur.fetchall()
            cur.execute(
                """
                SELECT a.lane_id, m.model_name
                FROM lane_model_aliases a
                JOIN models m ON m.model_id=a.model_id
                """
            )
            aliases = cur.fetchall()

    models_by_lane: dict[str, set[str]] = {}
    for r in policy + aliases:
        models_by_lane.setdefault(str(r["lane_id"]), set()).add(str(r["model_name"]))

    out = []
    for r in lanes:
        lane_id = str(r["lane_id"])
        known = sorted(models_by_lane.get(lane_id, set()))
        cm = (r.get("current_model_name") or "").strip()
        if cm and cm not in known:
            known.append(cm)
        out.append(
            {
                "host": str(r["host_name"]),
                "host_status": str(r["host_status"]),
                "lane_id": lane_id,
                "lane_type": str(r["lane_type"]),
                "base_url": str(r["base_url"]),
                "lane_status": str(r["lane_status"]),
                "current_model": cm or None,
                "known_models": known,
                "last_ok_at": r.get("last_ok_at").isoformat() if r.get("last_ok_at") else None,
                "last_probe_at": r.get("last_probe_at").isoformat() if r.get("last_probe_at") else None,
            }
        )
    return {"lanes": out}
