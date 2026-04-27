from __future__ import annotations

import unittest
from unittest import mock

from mesh_router import mw_control


class _Msg:
    def topic(self):  # noqa: ANN001
        return "mw.commands"

    def partition(self):  # noqa: ANN001
        return 0

    def offset(self):  # noqa: ANN001
        return 1


class _Producer:
    def __init__(self, *_args, **_kwargs):  # noqa: ANN001
        self.produced = []

    def produce(self, _topic, key=None, value=None, on_delivery=None):  # noqa: ANN001
        self.produced.append((key, value))
        if callable(on_delivery):
            on_delivery(None, _Msg())

    def poll(self, _t):  # noqa: ANN001
        return None

    def flush(self, _t):  # noqa: ANN001
        return 0


class _Consumer:
    def __init__(self, *_args, **_kwargs):  # noqa: ANN001
        return None

    def subscribe(self, _topics):  # noqa: ANN001
        return None

    def close(self):  # noqa: ANN001
        return None


class MWControlClientTimeoutTests(unittest.TestCase):
    def test_send_command_timeout_returns_pending(self) -> None:
        with mock.patch.object(mw_control, "Producer", _Producer), mock.patch.object(mw_control, "Consumer", _Consumer):
            client = mw_control.MeshWorkerCommandClient(
                bootstrap_servers="localhost:9092",
                commands_topic="mw.commands",
                responses_topic="mw.responses",
                client_id="test",
            )

            def _raise_timeout(*_args, **_kwargs):  # noqa: ANN001
                raise mw_control.MWControlTimeout(
                    "timed out waiting for MeshWorker response for request_id=req-1",
                    request_id="req-1",
                    timeout_seconds=7,
                )

            client._wait_for_response = _raise_timeout  # type: ignore[method-assign]

            result = client.send_command(
                host_id="static-deskix",
                message_type="activate_profile",
                payload={"profile_id": "split_default"},
                request_id="req-1",
                wait=True,
                timeout_seconds=7,
            )

        self.assertTrue(result.get("ok"))
        self.assertTrue(result.get("pending"))
        self.assertEqual(result.get("request_id"), "req-1")
        self.assertEqual(result.get("timeout_seconds"), 7)

    def test_send_command_missing_payload_ok_fails_closed(self) -> None:
        with mock.patch.object(mw_control, "Producer", _Producer), mock.patch.object(mw_control, "Consumer", _Consumer):
            client = mw_control.MeshWorkerCommandClient(
                bootstrap_servers="localhost:9092",
                commands_topic="mw.commands",
                responses_topic="mw.responses",
                client_id="test",
            )
            client._wait_for_response = lambda *_args, **_kwargs: {"payload": {"result": {"accepted": True}}}  # type: ignore[method-assign]

            result = client.send_command(
                host_id="static-deskix",
                message_type="load_model",
                payload={"lane_id": "gpu", "model_name": "qwen3.5-4b"},
                request_id="req-2",
                wait=True,
            )

        self.assertFalse(result.get("ok"))
        self.assertEqual(result.get("result"), {"accepted": True})


if __name__ == "__main__":
    unittest.main()
