from __future__ import annotations

import argparse
import json
import sys
import time

import httpx


def main() -> int:
    p = argparse.ArgumentParser(prog="mw_command_smoke")
    p.add_argument("--mr", default="http://mesh-router.example:4010", help="Mesh-Router base URL")
    p.add_argument("--host-id", required=True, help="MW host_id (e.g. worker-a)")
    p.add_argument("--command", required=True, help="activate_profile|load_model|health_probe|...")
    p.add_argument("--payload", default="{}", help="JSON payload for the command")
    p.add_argument("--timeout-seconds", type=int, default=None, help="Override MW wait timeout for this command")
    p.add_argument("--follow", action="store_true", help="If command returns pending/202, poll status until terminal or timeout")
    p.add_argument("--follow-timeout-s", type=int, default=180, help="Max seconds to poll when --follow is set")
    p.add_argument("--follow-interval-s", type=int, default=2, help="Poll interval seconds when --follow is set")
    p.add_argument("--print-http", action="store_true", help="Print HTTP status + Location/Retry-After headers to stderr")
    args = p.parse_args()

    try:
        payload = json.loads(args.payload)
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
    except Exception as exc:
        print(f"invalid --payload: {exc}", file=sys.stderr)
        return 2

    url = f"{args.mr.rstrip('/')}/api/mw/commands"
    body = {"host_id": args.host_id, "message_type": args.command, "payload": payload, "wait": True}
    if args.timeout_seconds is not None:
        body["timeout_seconds"] = int(args.timeout_seconds)
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=body)
        if args.print_http:
            loc = resp.headers.get("location")
            retry_after = resp.headers.get("retry-after")
            print(f"HTTP {resp.status_code}", file=sys.stderr)
            if loc:
                print(f"Location: {loc}", file=sys.stderr)
            if retry_after:
                print(f"Retry-After: {retry_after}", file=sys.stderr)
        if resp.status_code >= 400:
            print(resp.text, file=sys.stderr)
            return 2
        print(resp.text)
        is_pending = resp.status_code == 202
        if not is_pending:
            try:
                is_pending = bool(resp.json().get("pending"))
            except Exception:
                is_pending = False
        if is_pending:
            try:
                rid = str(resp.json().get("request_id") or "").strip()
            except Exception:
                rid = ""
            if rid:
                print(f"NOTE: pending; poll: {args.mr.rstrip('/')}/api/mw/commands/{rid}", file=sys.stderr)
        if args.follow and is_pending:
            try:
                rid = str(resp.json().get("request_id") or "").strip()
            except Exception:
                rid = ""
            if not rid:
                print("pending response missing request_id; cannot follow", file=sys.stderr)
                return 2
            deadline = time.monotonic() + max(1, int(args.follow_timeout_s))
            status_url = f"{args.mr.rstrip('/')}/api/mw/commands/{rid}"
            while time.monotonic() < deadline:
                time.sleep(max(1, int(args.follow_interval_s)))
                status = client.get(status_url)
                if status.status_code >= 400:
                    print(status.text, file=sys.stderr)
                    return 2
                print(status.text)
                payload_obj = status.json()
                if not payload_obj.get("found"):
                    continue
                state = str(payload_obj.get("status") or "")
                if state in {"completed", "failed", "rejected", "cancelled"}:
                    return 0 if payload_obj.get("ok") is not False else 2
            print(f"follow timeout after {args.follow_timeout_s}s; request_id={rid}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
