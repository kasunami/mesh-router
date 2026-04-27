from __future__ import annotations

from fastapi.testclient import TestClient

from mesh_router import app as app_module


def test_healthz_includes_revision() -> None:
    app_module.settings.allow_dev_secrets = True
    client = TestClient(app_module.app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "revision" in body


def test_health_dependencies_shape(monkeypatch) -> None:  # noqa: ANN001
    app_module.settings.allow_dev_secrets = True
    class _Conn:
        def __enter__(self):  # noqa: ANN001
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def execute(self, *_args, **_kwargs):  # noqa: ANN001
            return None

    class _Db:
        def connect(self):  # noqa: ANN001
            return _Conn()

    monkeypatch.setattr(app_module, "db", _Db())
    monkeypatch.setattr(app_module, "mw_state_db", _Db())

    client = TestClient(app_module.app)
    resp = client.get("/health/dependencies")
    assert resp.status_code == 200
    body = resp.json()
    assert "dependencies" in body
    assert body["dependencies"]["database"]["ok"] is True
    assert body["dependencies"]["mw_state_database"]["ok"] is True
