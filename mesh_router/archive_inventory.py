from __future__ import annotations

from .schemas import ArchiveInventoryScanRequest
from .scanner_utils import scan_model_root


def build_archive_inventory_payload(
    root_path: str, 
    archive_id: str, 
    provider: str
) -> ArchiveInventoryScanRequest:
    """Build the full archive inventory scan payload."""
    artifacts = scan_model_root(root_path)

    return ArchiveInventoryScanRequest(
        archive_id=archive_id,
        provider=provider,
        root_path=root_path,
        artifacts=artifacts,
        scan_details={
            "storage_scope": "archive",
            "artifact_count": len(artifacts),
        },
    )


if __name__ == "__main__":
    import json
    import sys
    import os

    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <archive_root> <archive_id> [provider]")
        sys.exit(1)

    root_path = sys.argv[1]
    archive_id = sys.argv[2]
    provider = sys.argv[3] if len(sys.argv) > 3 else "unknown"

    if not os.path.exists(root_path):
        print(f"Error: Path {root_path} does not exist.")
        sys.exit(1)

    payload = build_archive_inventory_payload(root_path, archive_id, provider)
    print(json.dumps(payload.model_dump(), indent=2))
