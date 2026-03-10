from __future__ import annotations

import argparse
import threading

import uvicorn

from .app import app
from .probe import run_forever as probe_forever
from .sync import run_forever


def main() -> int:
    parser = argparse.ArgumentParser(prog="mesh-router")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=4010, type=int)
    parser.add_argument("--no-sync", action="store_true", help="Disable worker-lane sync loop")
    parser.add_argument("--no-probe", action="store_true", help="Disable mesh-router health probe loop")
    args = parser.parse_args()

    if not args.no_sync:
        t = threading.Thread(target=run_forever, name="mesh-router-sync", daemon=True)
        t.start()

    if not args.no_probe:
        t2 = threading.Thread(target=probe_forever, name="mesh-router-probe", daemon=True)
        t2.start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
