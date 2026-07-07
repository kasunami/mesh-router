from __future__ import annotations

from mesh_router.db import _migration_directory


def test_migration_directory_contains_schema_files() -> None:
    migration_directory = _migration_directory()

    assert migration_directory.is_dir()
    assert (migration_directory / "001_init.sql").is_file()
