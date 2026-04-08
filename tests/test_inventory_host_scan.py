from __future__ import annotations

import unittest

from mesh_router import app as app_module


class _FakeCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object | None]] = []
        self.fetchone_rows = [{"model_id": "model-1"}, {"artifact_id": "artifact-1"}]

    def execute(self, sql, params=None):  # noqa: ANN001
        self.calls.append((str(sql), params))

    def fetchone(self):  # noqa: ANN001
        if self.fetchone_rows:
            return self.fetchone_rows.pop(0)
        return {"artifact_id": "artifact-x"}


class HostScanIngestTests(unittest.TestCase):
    def test_ingest_artifacts_marks_absent_only_within_root_scope(self) -> None:
        cur = _FakeCursor()
        artifact = type(
            "Artifact",
            (),
            {
                "name": "flux1-schnell-Q4_K_S",
                "path": "/srv/ai_models/image-models/flux-schnell/flux1-schnell-Q4_K_S.gguf",
                "size_bytes": 123,
                "format": "gguf",
                "checksum": None,
            },
        )()

        app_module._ingest_artifacts(  # type: ignore[attr-defined]
            cur,
            host_id="pupix1",
            artifacts=[artifact],
            storage_scope="local",
            storage_provider="local",
            root_path="/srv/ai_models/image-models",
        )

        absent_sql, absent_params = cur.calls[-1]
        assert "local_path=%s OR local_path LIKE %s" in absent_sql
        assert absent_params[0] == "pupix1"
        assert absent_params[1] == "local"
        assert absent_params[2] == "/srv/ai_models/image-models"
        assert absent_params[3] == "/srv/ai_models/image-models/%"
        assert absent_params[4] == ["/srv/ai_models/image-models/flux-schnell/flux1-schnell-Q4_K_S.gguf"]


if __name__ == "__main__":
    unittest.main()
