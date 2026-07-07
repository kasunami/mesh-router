#!/usr/bin/env python3
"""Print configured MeshRouter lanes and model artifacts as JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import psycopg
from psycopg.rows import dict_row


DEFAULT_DATABASE_URL = "postgresql://username:password@localhost:5432/mesh_router"


def load_lane_info(database_url: str) -> dict[str, list[dict[str, Any]]]:
    """Load active lane and model-artifact inventory from MeshRouter's database."""
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT h.host_id, h.host_name, h.mgmt_ssh_host,
                       l.lane_id, l.lane_name, l.lane_type, l.base_url
                FROM hosts h
                JOIN lanes l ON h.host_id = l.host_id
                WHERE h.status != 'offline' AND l.status != 'offline'
                ORDER BY h.host_name, l.lane_name
                """
            )
            lanes = list(cur.fetchall())

            cur.execute(
                """
                SELECT hma.host_id, m.model_id, m.model_name, m.format,
                       hma.local_path
                FROM host_model_artifacts hma
                JOIN models m ON hma.model_id = m.model_id
                WHERE hma.present = true
                ORDER BY hma.host_id, m.model_name, hma.local_path
                """
            )
            models = list(cur.fetchall())

    return {"lanes": lanes, "models": models}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print active MeshRouter lanes and available model artifacts as JSON."
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("MESH_ROUTER_DATABASE_URL", DEFAULT_DATABASE_URL),
        help="PostgreSQL DSN (defaults to MESH_ROUTER_DATABASE_URL)",
    )
    args = parser.parse_args(argv)

    try:
        payload = load_lane_info(args.database_url)
    except psycopg.Error as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
