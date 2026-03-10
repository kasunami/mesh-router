import psycopg
from psycopg.rows import dict_row
import json
import os

dsn = os.environ.get("MESH_ROUTER_DATABASE_URL", "postgresql://username:password@localhost:5432/mesh_router")

try:
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            # Get hosts and lanes
            cur.execute("""
                SELECT h.host_id, h.host_name, h.mgmt_ssh_host, l.lane_id, l.lane_name, l.lane_type, l.base_url
                FROM hosts h
                JOIN lanes l ON h.host_id = l.host_id
                WHERE h.status != 'offline' AND l.status != 'offline'
            """)
            lanes = cur.fetchall()
            
            # Get models available on hosts
            cur.execute("""
                SELECT hma.host_id, m.model_id, m.model_name, m.format, hma.local_path
                FROM host_model_artifacts hma
                JOIN models m ON hma.model_id = m.model_id
                WHERE hma.present = true
            """)
            models = cur.fetchall()

            data = {
                "lanes": lanes,
                "models": models
            }
            print(json.dumps(data, indent=2))
except Exception as e:
    print(json.dumps({"error": str(e)}))
