from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .config import settings


@dataclass(frozen=True)
class MeshBenchLease:
    lease_id: int
    lease_token: str
    proxy_base_url: str
    expires_at: str | None = None


class MeshBenchClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or settings.meshbench_base_url).rstrip("/")

    def acquire(self, *, worker_id: str, base_url: str, model: str, owner: str, job_type: str, ttl_seconds: int) -> MeshBenchLease:
        payload = {
            "worker_id": worker_id,
            "base_url": base_url,
            "model": model,
            "owner": owner,
            "job_type": job_type,
            "ttl_seconds": ttl_seconds,
        }
        with httpx.Client(timeout=10.0) as client:
            r = client.post(f"{self.base_url}/api/worker-leases/acquire", json=payload)
            if r.status_code == 409:
                raise RuntimeError("lane busy")
            r.raise_for_status()
            data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"lease acquire failed: {data}")
        return MeshBenchLease(
            lease_id=int(data["lease_id"]),
            lease_token=str(data["lease_token"]),
            proxy_base_url=str(data.get("proxy_base_url") or self.base_url),
            expires_at=str(data.get("expires_at")) if data.get("expires_at") else None,
        )

    def release(self, *, lease_id: int, outcome: str = "released", status_after: str = "ready") -> None:
        payload = {"lease_id": lease_id, "outcome": outcome, "status_after": status_after}
        with httpx.Client(timeout=10.0) as client:
            r = client.post(f"{self.base_url}/api/worker-leases/release", json=payload)
            # Best-effort release.
            if r.status_code >= 500:
                return

    def proxy_chat_completions(self, *, lease_token: str, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=600.0) as client:
            r = client.post(
                f"{self.base_url}/api/worker-proxy/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {lease_token}"},
            )
            try:
                data = r.json()
            except Exception:
                data = {"raw": r.text}
            if r.status_code >= 400:
                raise RuntimeError(f"meshbench proxy http_{r.status_code}: {data}")
            return data
