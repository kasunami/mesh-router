#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx


CHECKS = {
    "chat",
    "chat-stream",
    "embeddings",
    "image",
    "pin-worker",
    "pin-lane",
    "mw-command",
    "inventory",
}


@dataclass
class Result:
    name: str
    ok: bool
    status: str
    details: dict[str, Any]


def _csv(value: str) -> list[str]:
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not items or "all" in items:
        return sorted(CHECKS)
    unknown = sorted(set(items) - CHECKS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown check(s): {', '.join(unknown)}")
    return items


def _post(client: httpx.Client, url: str, payload: dict[str, Any]) -> tuple[int, dict[str, str], Any]:
    resp = client.post(url, json=payload)
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text[:500]}
    return resp.status_code, dict(resp.headers), body


def _mesh_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower().startswith("x-mesh-")
    }


def _chat_payload(model: str, prompt: str, *, stream: bool) -> dict[str, Any]:
    return {
        "model": model,
        "stream": stream,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 32,
    }


def _result(name: str, ok: bool, status: str, **details: Any) -> Result:
    return Result(name=name, ok=ok, status=status, details={k: v for k, v in details.items() if v is not None})


def run_inventory(client: httpx.Client, base_url: str) -> Result:
    resp = client.get(f"{base_url}/api/inventory")
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text[:500]}
    return _result("inventory", resp.status_code == 200, f"http_{resp.status_code}", body_keys=sorted(body.keys()) if isinstance(body, dict) else [])


def run_chat(client: httpx.Client, base_url: str, model: str, prompt: str) -> Result:
    code, headers, body = _post(client, f"{base_url}/v1/chat/completions", _chat_payload(model, prompt, stream=False))
    ok = 200 <= code < 300 and isinstance(body, dict) and bool(body.get("choices"))
    return _result("chat", ok, f"http_{code}", mesh_headers=_mesh_headers(headers), body_keys=sorted(body.keys()) if isinstance(body, dict) else [])


def run_chat_stream(client: httpx.Client, base_url: str, model: str, prompt: str) -> Result:
    chunks = 0
    mesh_headers: dict[str, str] = {}
    try:
        with client.stream("POST", f"{base_url}/v1/chat/completions", json=_chat_payload(model, prompt, stream=True)) as resp:
            mesh_headers = _mesh_headers(dict(resp.headers))
            if resp.status_code < 200 or resp.status_code >= 300:
                return _result("chat-stream", False, f"http_{resp.status_code}", mesh_headers=mesh_headers, body=resp.read().decode("utf-8", errors="replace")[:500])
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    chunks += 1
                if line.strip() == "data: [DONE]":
                    break
    except Exception as exc:
        return _result("chat-stream", False, "error", error=str(exc))
    return _result("chat-stream", chunks > 0, "ok", chunks=chunks, mesh_headers=mesh_headers)


def run_embeddings(client: httpx.Client, base_url: str, model: str, prompt: str) -> Result:
    code, headers, body = _post(client, f"{base_url}/v1/embeddings", {"model": model, "input": [prompt]})
    ok = 200 <= code < 300 and isinstance(body, dict)
    return _result("embeddings", ok, f"http_{code}", mesh_headers=_mesh_headers(headers), body_keys=sorted(body.keys()) if isinstance(body, dict) else [])


def run_image(client: httpx.Client, base_url: str, model: str, prompt: str) -> Result:
    payload = {"model": model, "prompt": prompt, "n": 1, "size": "512x512", "response_format": "b64_json"}
    code, headers, body = _post(client, f"{base_url}/v1/images/generations", payload)
    ok = 200 <= code < 300 and isinstance(body, dict) and bool(body.get("data"))
    return _result("image", ok, f"http_{code}", mesh_headers=_mesh_headers(headers), body_keys=sorted(body.keys()) if isinstance(body, dict) else [])


def run_pin_worker(client: httpx.Client, base_url: str, model: str, prompt: str, worker: str) -> Result:
    payload = _chat_payload(model, prompt, stream=False)
    payload["mesh_pin_worker"] = worker
    code, headers, body = _post(client, f"{base_url}/v1/chat/completions", payload)
    ok = 200 <= code < 300 and headers.get("x-mesh-worker-id") == worker
    return _result("pin-worker", ok, f"http_{code}", mesh_headers=_mesh_headers(headers), body_keys=sorted(body.keys()) if isinstance(body, dict) else [])


def run_pin_lane(client: httpx.Client, base_url: str, model: str, prompt: str, lane_id: str) -> Result:
    payload = _chat_payload(model, prompt, stream=False)
    payload["mesh_pin_lane_id"] = lane_id
    code, headers, body = _post(client, f"{base_url}/v1/chat/completions", payload)
    ok = 200 <= code < 300 and headers.get("x-mesh-lane-id") == lane_id
    return _result("pin-lane", ok, f"http_{code}", mesh_headers=_mesh_headers(headers), body_keys=sorted(body.keys()) if isinstance(body, dict) else [])


def run_mw_command(client: httpx.Client, base_url: str, host_id: str, lane_id: str) -> Result:
    payload = {"host_id": host_id, "message_type": "health_probe", "payload": {"lane_id": lane_id}, "wait": True, "timeout_seconds": 10}
    code, headers, body = _post(client, f"{base_url}/api/mw/commands", payload)
    details: dict[str, Any] = {"mesh_headers": _mesh_headers(headers), "body": body if isinstance(body, dict) else {}}
    if code == 202 and isinstance(body, dict) and body.get("request_id"):
        request_id = str(body["request_id"])
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status = client.get(f"{base_url}/api/mw/commands/{request_id}")
            try:
                status_body = status.json()
            except Exception:
                status_body = {"raw": status.text[:500]}
            details["poll_status"] = status.status_code
            details["poll_body"] = status_body
            if isinstance(status_body, dict) and status_body.get("status") in {"ready", "completed", "failed", "rejected", "cancelled"}:
                return _result("mw-command", bool(status_body.get("ok")), str(status_body.get("status")), **details)
            time.sleep(2)
        return _result("mw-command", False, "poll_timeout", **details)
    ok = 200 <= code < 300 and isinstance(body, dict) and bool(body.get("ok"))
    return _result("mw-command", ok, f"http_{code}", **details)


def main() -> int:
    parser = argparse.ArgumentParser(description="Certify Mesh Router live behavior.")
    parser.add_argument("--base-url", default="http://mesh-router.example:4010")
    parser.add_argument("--mode", choices=["dry-run", "live"], default="dry-run")
    parser.add_argument("--checks", type=_csv, default=sorted(CHECKS))
    parser.add_argument("--chat-model", default="")
    parser.add_argument("--embeddings-model", default="")
    parser.add_argument("--image-model", default="")
    parser.add_argument("--pin-worker", default="")
    parser.add_argument("--pin-lane-id", default="")
    parser.add_argument("--mw-host-id", default="")
    parser.add_argument("--mw-lane-id", default="")
    parser.add_argument("--prompt", default="Reply with ok.")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    selected = args.checks if isinstance(args.checks, list) else _csv(str(args.checks))
    if args.mode == "dry-run":
        print(json.dumps({"ok": True, "mode": "dry-run", "base_url": args.base_url, "checks": selected}, indent=2))
        return 0

    required_by_check = {
        "chat": ["chat_model"],
        "chat-stream": ["chat_model"],
        "embeddings": ["embeddings_model"],
        "image": ["image_model"],
        "pin-worker": ["chat_model", "pin_worker"],
        "pin-lane": ["chat_model", "pin_lane_id"],
        "mw-command": ["mw_host_id", "mw_lane_id"],
    }
    missing = sorted({field for check in selected for field in required_by_check.get(check, []) if not getattr(args, field)})
    if missing:
        print(json.dumps({"ok": False, "error": "missing required live arguments", "missing": missing}, indent=2), file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    results: list[Result] = []
    with httpx.Client(timeout=httpx.Timeout(args.timeout_s, read=args.timeout_s)) as client:
        for check in selected:
            if check == "inventory":
                results.append(run_inventory(client, base_url))
            elif check == "chat":
                results.append(run_chat(client, base_url, args.chat_model, args.prompt))
            elif check == "chat-stream":
                results.append(run_chat_stream(client, base_url, args.chat_model, args.prompt))
            elif check == "embeddings":
                results.append(run_embeddings(client, base_url, args.embeddings_model, args.prompt))
            elif check == "image":
                results.append(run_image(client, base_url, args.image_model, args.prompt))
            elif check == "pin-worker":
                results.append(run_pin_worker(client, base_url, args.chat_model, args.prompt, args.pin_worker))
            elif check == "pin-lane":
                results.append(run_pin_lane(client, base_url, args.chat_model, args.prompt, args.pin_lane_id))
            elif check == "mw-command":
                results.append(run_mw_command(client, base_url, args.mw_host_id, args.mw_lane_id))

    summary = {
        "ok": all(r.ok for r in results),
        "base_url": base_url,
        "results": [r.__dict__ for r in results],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
