from __future__ import annotations

import unittest

from mesh_router import app as app_module


class ModelMatchingExactTests(unittest.TestCase):
    def test_exact_request_does_not_fuzzy_match_family_size(self) -> None:
        requested = "Qwen3.5-9B-MLX-4bit"
        self.assertTrue(app_module._is_exact_model_request(requested))  # type: ignore[attr-defined]

        self.assertTrue(app_module._model_request_matches_candidate(requested, "Qwen3.5-9B-MLX-4bit"))  # type: ignore[attr-defined]
        self.assertFalse(app_module._model_request_matches_candidate(requested, "Qwen3.5-9B-6bit"))  # type: ignore[attr-defined]
        self.assertFalse(
            app_module._model_request_matches_candidate(requested, "/Users/kasunami/models/Qwen3.5-9B-6bit")  # type: ignore[attr-defined]
        )

    def test_exact_request_matches_candidate_basename_for_local_paths(self) -> None:
        requested = "Qwen3.5-9B-6bit"
        self.assertTrue(app_module._is_exact_model_request(requested))  # type: ignore[attr-defined]
        self.assertTrue(
            app_module._model_request_matches_candidate(requested, "/Users/kasunami/models/Qwen3.5-9B-6bit")  # type: ignore[attr-defined]
        )


if __name__ == "__main__":
    unittest.main()

