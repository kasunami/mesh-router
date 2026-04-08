from __future__ import annotations

import unittest

from mesh_router import app as app_module


class _FakeCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, sql: str, params: tuple[object, ...]) -> None:
        self.calls.append((sql, params))


class LaneSuspensionTests(unittest.TestCase):
    def test_unsuspend_clears_matching_reason_for_suspended_lane(self) -> None:
        cur = _FakeCursor()

        app_module._set_lane_suspension(
            cur,
            lane_id="lane-1",
            suspended=False,
            reason="swap:abc:stopping_siblings",
        )

        sql, params = cur.calls[-1]
        self.assertIn("status IN ('offline', 'suspended')", sql)
        self.assertEqual(params, ("swap:abc:stopping_siblings", "swap:abc:stopping_siblings", "lane-1"))


if __name__ == "__main__":
    unittest.main()
