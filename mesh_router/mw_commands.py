from __future__ import annotations

import time
from typing import Any, Callable

from .db import mw_state_db
from .mw_control import MeshWorkerCommandClient


def fetch_mw_transition_status(request_id: str) -> dict[str, Any] | None:
    rid = str(request_id or "").strip()
    if not rid:
        return None
    with mw_state_db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT request_id, host_id, transition_type, status, current_phase, error_kind, error_message, updated_at
                FROM mw_transitions
                WHERE request_id::text=%s
                LIMIT 1
                """,
                (rid,),
            )
            row = cur.fetchone()
    return dict(row) if row else None


def wait_for_mw_transition_terminal(*, request_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + max(1, int(timeout_seconds))
    terminal = {"ready", "completed", "failed", "cancelled", "canceled", "rejected"}
    last_row: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        row = fetch_mw_transition_status(request_id)
        if row:
            last_row = row
            status = str(row.get("status") or "").strip().lower()
            if status in terminal:
                if status in {"ready", "completed"}:
                    return row
                raise RuntimeError(str(row.get("error_message") or row.get("error_kind") or f"MW transition {status}"))
        time.sleep(1)
    status_text = str((last_row or {}).get("status") or "not_found")
    raise RuntimeError(f"MW transition did not complete within {timeout_seconds}s (request_id={request_id}, status={status_text})")


def send_mw_command_require_ready(
    *,
    client_factory: Callable[[], MeshWorkerCommandClient],
    host_id: str,
    message_type: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    result = client_factory().send_command(
        host_id=host_id,
        message_type=message_type,
        payload=payload,
        wait=True,
        timeout_seconds=timeout_seconds,
    )
    if result.get("pending"):
        request_id = str(result.get("request_id") or "").strip()
        if not request_id:
            raise RuntimeError("MW command is pending but response did not include request_id")
        transition = wait_for_mw_transition_terminal(
            request_id=request_id,
            timeout_seconds=timeout_seconds,
        )
        return {**result, "ok": True, "pending": False, "transition": transition}
    if not bool(result.get("ok", False)):
        raise RuntimeError(str(result.get("error") or result.get("warning") or f"MW {message_type} failed"))
    return result
