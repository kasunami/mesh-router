from __future__ import annotations

import unittest
from unittest import mock

from mesh_router import router as router_module


class _Cur:
    def __enter__(self):  # noqa: ANN001
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class _Conn:
    def cursor(self):  # noqa: ANN001
        return _Cur()

    def __enter__(self):  # noqa: ANN001
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class _Db:
    def connect(self):  # noqa: ANN001
        return _Conn()


class PinLaneIdPlacementTests(unittest.TestCase):
    def test_pin_lane_id_fails_without_fallback(self) -> None:
        with mock.patch.object(router_module, "db", _Db()), mock.patch.object(router_module, "q", return_value=[]):
            with self.assertRaises(router_module.LanePlacementError) as ctx:
                router_module.pick_lane_for_model(model="qwen3.5-2b", pin_lane_id="missing-lane")
        self.assertEqual(getattr(ctx.exception, "status_code", None), 404)


if __name__ == "__main__":
    unittest.main()

