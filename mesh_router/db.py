from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from .config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Db:
    dsn: str

    @contextmanager
    def connect(self) -> Iterator[psycopg.Connection]:
        conn = psycopg.connect(self.dsn, row_factory=dict_row)
        try:
            yield conn
        finally:
            conn.close()


db = Db(settings.database_url)


def init_db() -> None:
    sql_dir = Path(__file__).resolve().parent.parent / "sql"
    if not sql_dir.exists() or not sql_dir.is_dir():
        logger.info("SQL directory not found at %s, skipping migrations", sql_dir)
        return

    sql_files = sorted(sql_dir.glob("*.sql"))
    if not sql_files:
        return

    logger.info("Running %d SQL migrations", len(sql_files))
    with db.connect() as conn:
        for sql_file in sql_files:
            try:
                conn.execute(sql_file.read_text())
            except Exception as e:
                logger.error("Failed to execute %s: %s", sql_file.name, e)
                raise
        conn.commit()


def q1(cur: psycopg.Cursor, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    cur.execute(sql, params)
    return cur.fetchone()


def q(cur: psycopg.Cursor, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    cur.execute(sql, params)
    return list(cur.fetchall())

