from mesh_router import db


def test_init_db_skips_migrations_when_runtime_setting_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr(db.settings, "auto_migrate_on_startup", False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("migration DDL must not run in this runtime")

    monkeypatch.setattr(db, "_apply_migrations", fail_if_called)

    db.init_db()
