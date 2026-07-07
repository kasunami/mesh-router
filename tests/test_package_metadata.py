from __future__ import annotations

from importlib.metadata import metadata


def test_distribution_declares_apache_license() -> None:
    assert metadata("mesh-router")["License-Expression"] == "Apache-2.0"
