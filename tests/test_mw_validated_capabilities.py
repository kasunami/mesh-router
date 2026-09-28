from __future__ import annotations

from datetime import UTC, datetime
import unittest

from mesh_router import app as app_module
from mesh_router.schemas import LaneModelCandidate
from mesh_router.viability import ViabilityLaneInfo


class MwValidatedCapabilitiesTests(unittest.TestCase):
    def test_mw_validated_local_candidate_is_persisted_for_catalog(self) -> None:
        class _Cursor:
            def __init__(self) -> None:
                self.calls: list[tuple[str, tuple[object, ...] | None]] = []

            def execute(self, sql, params=None):  # noqa: ANN001
                self.calls.append((sql, params))

        cur = _Cursor()
        candidate = LaneModelCandidate(
            model_name="qwen3.6-35b-a3b",
            tags=["mw-validated", "chat"],
            locality="local",
            artifact_path="/models/qwen3.6-35b-a3b.gguf",
        )
        app_module._persist_mw_validated_viability(
            cur,
            lane_id="lane-1",
            candidates_by_model={candidate.model_name: candidate},
            artifact_rows=[
                {
                    "artifact_id": "artifact-1",
                    "host_id": "host-1",
                    "model_id": "model-1",
                    "model_name": candidate.model_name,
                    "local_path": "/models/qwen3.6-35b-a3b.gguf",
                }
            ],
            host_id="host-1",
            local_model_root="/models",
        )

        self.assertEqual(len(cur.calls), 1)
        sql, params = cur.calls[0]
        self.assertIn("is_viable = true", sql)
        self.assertIn("tps_source = 'mw_validated'", sql)
        self.assertEqual(params, ("lane-1", "model-1", "artifact-1"))

    def test_mw_validated_candidate_overrides_stale_zero_tps_viability_gap(self) -> None:
        candidates: dict[str, LaneModelCandidate] = {}
        lane_row = {
            "lane_id": "lane-1",
            "lane_name": "cpu",
            "lane_type": "cpu",
            "backend_type": "llama",
            "current_model_name": "google_gemma-4-26B-A4B-it-Q4_K_M.gguf",
            "proxy_auth_metadata": {
                "control_plane": "mw",
                "mw_host_id": "worker-b",
                "mw_lane_id": "cpu",
            },
            "validated_candidates": [
                {
                    "canonical_id": "Qwen3.5-9B-Q4_K_M.gguf",
                    "backend_types": ["llama.cpp"],
                    "lane_ids": ["cpu"],
                    "max_ctx": 131072,
                    "tags": ["proven_runnable"],
                }
            ],
        }
        host_row = {"host_id": "host-1", "host_name": "Worker-B"}
        artifact_rows = [
            {
                "artifact_id": "artifact-1",
                "host_id": "host-1",
                "host_name": "Worker-B",
                "storage_provider": "local",
                "local_path": "/models/Qwen3.5-9B-Q4_K_M.gguf",
                "size_bytes": 5680522464,
                "model_id": "model-1",
                "model_name": "Qwen3.5-9B-Q4_K_M.gguf",
            }
        ]

        app_module._merge_mw_validated_capability_candidates(
            candidates_by_model=candidates,
            lane_row=lane_row,
            host_row=host_row,
            lane_info=ViabilityLaneInfo(lane_id="lane-1", lane_type="cpu", ram_budget_bytes=24 * 1024**3),
            artifact_rows=artifact_rows,
            local_model_root="/models",
        )

        candidate = candidates["Qwen3.5-9B-Q4_K_M.gguf"]
        self.assertEqual(candidate.locality, "local")
        self.assertEqual(candidate.artifact_path, "/models/Qwen3.5-9B-Q4_K_M.gguf")
        self.assertEqual(candidate.max_context_tokens, 131072)
        self.assertIn("mw-validated", candidate.tags)
        self.assertIn("qwen3.5:9b", candidate.tags)

    def test_model_tuning_profile_response_accepts_legacy_local_storage_scheme(self) -> None:
        row = {
            "tuning_profile_id": "profile-1",
            "host_id": "host-1",
            "host_name": "worker-f",
            "model_id": "model-1",
            "model_name": "google_gemma-4-E2B-it-Q4_K_M.gguf",
            "lane_id": "lane-1",
            "lane_name": "gpu",
            "lane_type": "gpu",
            "storage_scheme": "local",
            "settings": {},
            "cost_tier": "standard",
            "disables_sibling_lanes": False,
            "exclusive_host_resources": False,
            "evaluation_count": 1,
            "created_at": datetime.now(tz=UTC),
            "updated_at": datetime.now(tz=UTC),
        }

        profile = app_module._row_to_tuning_profile(row)

        self.assertEqual(profile.storage_scheme, "local")


if __name__ == "__main__":
    unittest.main()
