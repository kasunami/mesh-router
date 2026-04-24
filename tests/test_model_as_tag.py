from __future__ import annotations

import unittest
from unittest.mock import patch

from mesh_router import router as router_module
from mesh_router.router import LaneChoice, LanePlacementError


class ModelAsTagTests(unittest.TestCase):
    def test_model_string_can_be_a_tag_and_falls_back_to_tagged_models(self) -> None:
        calls: list[str] = []

        def _fake_single(**kwargs):  # noqa: ANN001
            calls.append(str(kwargs.get("model")))
            if kwargs.get("model") == "home-assistant-voice":
                raise RuntimeError("no READY lanes available serving requested model: home-assistant-voice")
            return LaneChoice(
                lane_id="lane-1",
                worker_id="Static-Deskix",
                base_url="http://10.0.0.99:11434",
                lane_type="gpu",
                backend_type="llama",
                current_model_name="Qwen3.5-9B-Q4_K_M.gguf",
                resolved_model_name=str(kwargs.get("model")),
            )

        with (
            patch.object(router_module, "_pick_lane_for_model_single", side_effect=_fake_single),
            patch.object(router_module, "_models_for_tag", return_value=["qwen3.5-9b", "falcon3-10b"]),
        ):
            choice = router_module.pick_lane_for_model(model="home-assistant-voice")

        self.assertEqual(choice.worker_id, "Static-Deskix")
        self.assertEqual(calls, ["home-assistant-voice", "qwen3.5-9b"])

    def test_tag_fallback_does_not_mask_lane_placement_errors(self) -> None:
        def _fake_single(**kwargs):  # noqa: ANN001
            raise LanePlacementError("pinned lane does not match requested backend", status_code=409)

        with (
            patch.object(router_module, "_pick_lane_for_model_single", side_effect=_fake_single),
            patch.object(router_module, "_models_for_tag", side_effect=AssertionError("_models_for_tag must not be called")),
        ):
            with self.assertRaises(LanePlacementError):
                router_module.pick_lane_for_model(model="home-assistant-voice", pin_lane_id="lane-123")


if __name__ == "__main__":
    unittest.main()

