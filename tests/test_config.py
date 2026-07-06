from __future__ import annotations

import unittest

from mesh_router.config import Settings, validate_runtime_settings


def test_settings_loads_local_dotenv(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.delenv("MESH_ROUTER_DEFAULT_OWNER", raising=False)
    (tmp_path / ".env").write_text("MESH_ROUTER_DEFAULT_OWNER=dotenv-owner\n")
    monkeypatch.chdir(tmp_path)

    cfg = Settings()

    assert cfg.default_owner == "dotenv-owner"


class ConfigValidationTests(unittest.TestCase):
    def test_placeholder_secrets_fail_closed(self) -> None:
        cfg = Settings(
            lease_token_secret="replace-with-random-secret",
            swap_auth_token="replace-with-worker-swap-token",
        )
        with self.assertRaises(RuntimeError):
            validate_runtime_settings(cfg)

    def test_dev_secret_override_allows_placeholders(self) -> None:
        cfg = Settings(
            lease_token_secret="replace-with-random-secret",
            swap_auth_token="replace-with-worker-swap-token",
            allow_dev_secrets=True,
        )
        validate_runtime_settings(cfg)

    def test_real_secrets_pass(self) -> None:
        cfg = Settings(
            lease_token_secret="test-lease-secret",
            swap_auth_token="test-swap-secret",
        )
        validate_runtime_settings(cfg)


if __name__ == "__main__":
    unittest.main()
