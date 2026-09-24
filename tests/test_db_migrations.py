from types import SimpleNamespace

from mesh_router import cli, db


def test_runtime_startup_skips_migrations_when_disabled() -> None:
    settings = SimpleNamespace(auto_migrate_on_startup=False)

    def fail_if_called() -> None:
        raise AssertionError("runtime startup migration must be skipped")

    assert cli._run_startup_migrations(settings, fail_if_called) is False


def test_runtime_startup_runs_migrations_when_enabled() -> None:
    settings = SimpleNamespace(auto_migrate_on_startup=True)
    calls: list[str] = []

    assert cli._run_startup_migrations(settings, lambda: calls.append("migrated")) is True
    assert calls == ["migrated"]


def test_direct_init_db_still_runs_migrations_when_startup_opt_out_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr(db.settings, "auto_migrate_on_startup", False)
    monkeypatch.setattr(db.settings, "mw_state_database_url", None)
    calls: list[tuple[object, str]] = []

    monkeypatch.setattr(db, "_apply_migrations", lambda target, label: calls.append((target, label)))

    db.init_db()

    assert calls == [(db.db, "primary")]
