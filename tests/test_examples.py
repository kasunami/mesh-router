from __future__ import annotations

import json
from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_json_examples_parse_and_use_public_placeholders() -> None:
    json_examples = sorted(EXAMPLES_DIR.glob("*.json"))

    assert [path.name for path in json_examples] == [
        "chat-completion-request.json",
        "explicit-lane-pinning-request.json",
    ]
    payloads = [json.loads(path.read_text()) for path in json_examples]
    assert all(payload["model"] == "llama-3.1-8b-instruct" for payload in payloads)
    assert payloads[1]["mesh_pin_lane_id"] == "lane-gpu-a"


def test_http_transcripts_cover_route_metadata_and_fail_closed_behavior() -> None:
    success = (EXAMPLES_DIR / "route-metadata-response.txt").read_text()
    failure = (EXAMPLES_DIR / "fail-closed-exact-lane.txt").read_text()

    assert "x-mesh-worker-id: worker-gpu-01" in success
    assert "x-mesh-lane-id: lane-gpu-a" in success
    assert "HTTP/1.1 409 Conflict" in failure
    assert '"detail": "pinned lane is not ready"' in failure
