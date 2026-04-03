from __future__ import annotations

import argparse
import sys

import httpx


def main() -> int:
    p = argparse.ArgumentParser(prog="mw_command_smoke")
    p.add_argument("--mr", default="http://10.0.1.47:4010", help="Mesh-Router base URL")
    p.add_argument("--host-id", required=True, help="MW host_id (e.g. static-deskix)")
    p.add_argument("--command", required=True, help="activate_profile|load_model|health_probe|...")
    p.add_argument("--payload", default="{}", help="JSON payload for the command")
    args = p.parse_args()

    try:
        import json

        payload = json.loads(args.payload)
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
    except Exception as exc:
        print(f"invalid --payload: {exc}", file=sys.stderr)
        return 2

    url = f"{args.mr.rstrip('/')}/api/mw/commands"
    body = {"host_id": args.host_id, "message_type": args.command, "payload": payload, "wait": True}
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=body)
        if resp.status_code >= 400:
            print(resp.text, file=sys.stderr)
            return 2
        print(resp.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

