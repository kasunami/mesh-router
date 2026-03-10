from __future__ import annotations

import os
import socket
import psutil
from typing import Any, Optional

from .schemas import HostInventoryScanRequest
from .scanner_utils import scan_model_root


def get_host_facts() -> dict[str, Any]:
    """Capture basic host memory and runtime facts."""
    mem = psutil.virtual_memory()
    facts = {
        "hostname": socket.gethostname(),
        "total_ram_gb": round(mem.total / (1024**3), 2),
        "available_ram_gb": round(mem.available / (1024**3), 2),
        "platform": os.uname().sysname,
        "release": os.uname().release,
    }
    # Placeholder for GPU facts (VRAM/Unified Memory)
    facts["gpu"] = None
    return facts


def build_inventory_payload(root_path: str, host_id: Optional[str] = None) -> HostInventoryScanRequest:
    """Build the full host inventory scan payload."""
    host_facts = get_host_facts()
    if not host_id:
        host_id = host_facts["hostname"]

    artifacts = scan_model_root(root_path)

    return HostInventoryScanRequest(
        host_id=host_id,
        root_path=root_path,
        artifacts=artifacts,
        host_facts=host_facts,
        scan_details={
            "artifact_count": len(artifacts),
        },
    )


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_root>")
        sys.exit(1)

    payload = build_inventory_payload(sys.argv[1])
    print(json.dumps(payload.model_dump(), indent=2))
