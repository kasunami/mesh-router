from __future__ import annotations

import argparse
import sys

import httpx


def main() -> int:
    p = argparse.ArgumentParser(prog="mw_stream_smoke")
    p.add_argument("--mr", default="http://mesh-router.example:4010", help="Mesh-Router base URL")
    p.add_argument("--model", required=True, help="Model name")
    p.add_argument("--prompt", default="Hello from MW stream smoke test.", help="Prompt")
    p.add_argument("--lines", type=int, default=50, help="Max SSE lines to print")
    p.add_argument("--read-timeout-s", type=float, default=600.0, help="HTTP read timeout (seconds)")
    args = p.parse_args()

    url = f"{args.mr.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": args.model,
        "stream": True,
        "messages": [{"role": "user", "content": args.prompt}],
    }
    timeout = httpx.Timeout(60.0, read=float(args.read_timeout_s))
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, json=payload) as resp:
            if resp.status_code >= 400:
                body = resp.read()
                text = body.decode("utf-8", errors="replace") if isinstance(body, (bytes, bytearray)) else str(body)
                print(
                    f"status={resp.status_code} content-type={resp.headers.get('content-type')}\n{text}",
                    file=sys.stderr,
                )
                return 2
            print(f"status={resp.status_code} content-type={resp.headers.get('content-type')}")
            printed = 0
            for raw in resp.iter_lines():
                if raw is None:
                    continue
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
                if line:
                    print(line)
                    printed += 1
                if printed >= args.lines:
                    break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
