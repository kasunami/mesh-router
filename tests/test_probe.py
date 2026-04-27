from __future__ import annotations

import unittest

from mesh_router import probe as probe_module


class ProbeRecoveryTests(unittest.TestCase):
    def test_recoverable_terminal_swap_failure_is_detected(self) -> None:
        self.assertTrue(probe_module._is_recoverable_terminal_swap_suspension("swap:abc:failed"))

    def test_recoverable_terminal_swap_cancel_is_detected(self) -> None:
        self.assertTrue(probe_module._is_recoverable_terminal_swap_suspension("swap:abc:canceled"))

    def test_non_terminal_swap_suspension_is_not_detected(self) -> None:
        self.assertFalse(probe_module._is_recoverable_terminal_swap_suspension("swap:abc:stopping_siblings"))

    def test_non_swap_suspension_is_not_detected(self) -> None:
        self.assertFalse(probe_module._is_recoverable_terminal_swap_suspension("dualboot_other_side_active: packpup1"))

    def test_mw_managed_lane_is_detected(self) -> None:
        self.assertTrue(
            probe_module._lane_is_mw_managed(
                {"proxy_auth_metadata": {"control_plane": "mw", "mw_host_id": "worker-a", "mw_lane_id": "gpu"}}
            )
        )

    def test_non_mw_lane_is_not_detected(self) -> None:
        self.assertFalse(probe_module._lane_is_mw_managed({"proxy_auth_metadata": {"control_plane": "legacy"}}))


if __name__ == "__main__":
    unittest.main()
