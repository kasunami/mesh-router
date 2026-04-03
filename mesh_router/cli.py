from __future__ import annotations

import argparse
import sys
import threading



def main() -> int:
    parser = argparse.ArgumentParser(prog="mesh-router")

    # Subcommand or top-level flags.
    if len(sys.argv) > 1 and sys.argv[1] == "inventory":
        from .worker_inventory import build_inventory_payload
        import json

        inv_parser = argparse.ArgumentParser(prog="mesh-router inventory")
        inv_parser.add_argument("root", help="Path to local model root")
        inv_parser.add_argument("--host-id", help="Override host ID (defaults to hostname)")
        inv_args = inv_parser.parse_args(sys.argv[2:])

        payload = build_inventory_payload(inv_args.root, host_id=inv_args.host_id)
        print(json.dumps(payload.model_dump(), indent=2))
        return 0

    if len(sys.argv) > 1 and sys.argv[1] == "archive-inventory":
        from .archive_inventory import build_archive_inventory_payload
        import json

        inv_parser = argparse.ArgumentParser(prog="mesh-router archive-inventory")
        inv_parser.add_argument("root", help="Path to archive model root")
        inv_parser.add_argument("archive_id", help="Archive identity")
        inv_parser.add_argument("--provider", default="unknown", help="Archive provider name")
        inv_args = inv_parser.parse_args(sys.argv[2:])

        payload = build_archive_inventory_payload(
            inv_args.root,
            archive_id=inv_args.archive_id,
            provider=inv_args.provider,
        )
        print(json.dumps(payload.model_dump(), indent=2))
        return 0

    # Handle serve command or flags (backward compatible).
    start_idx = 1
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        start_idx = 2

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=4010, type=int)
    parser.add_argument("--no-sync", action="store_true", help="Disable worker-lane sync loop")
    parser.add_argument("--no-probe", action="store_true", help="Disable mesh-router health probe loop")
    parser.add_argument("--no-mw-consume", action="store_true", help="Disable MW Kafka state/heartbeat consumer")
    args = parser.parse_args(sys.argv[start_idx:])

    from .db import init_db

    # Run database migrations on startup
    init_db()

    import uvicorn
    from .app import app
    from .probe import run_forever as probe_forever
    from .sync import run_forever

    if not args.no_sync:
        t = threading.Thread(target=run_forever, name="mesh-router-sync", daemon=True)
        t.start()

    if not args.no_probe:
        t2 = threading.Thread(target=probe_forever, name="mesh-router-probe", daemon=True)
        t2.start()

    if not args.no_mw_consume:
        from .mw_consumer import run_forever as mw_consume_forever

        t3 = threading.Thread(target=mw_consume_forever, name="mesh-router-mw-consumer", daemon=True)
        t3.start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
