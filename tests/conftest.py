from __future__ import annotations

import pytest

from mesh_router.config import settings


@pytest.fixture(autouse=True)
def allow_dev_secrets_for_tests() -> None:
    settings.allow_dev_secrets = True
