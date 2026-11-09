"""Central SQLite schema for git triage.

Run once (or anytime) to create/refresh tables:
    uv run python -m triage.db
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "git_triage.db"

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS issues (
        number      INTEGER PRIMARY KEY,
        title       TEXT,
        body        TEXT,
        labels      TEXT,
        state       TEXT,
        created_at  TEXT,
        fetched_at  TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id          TEXT PRIMARY KEY,
        created_at  TEXT,
        total       INTEGER,
        config      TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS items (
        job_id          TEXT NOT NULL,
        issue_number    INTEGER NOT NULL,
        status          TEXT NOT NULL DEFAULT 'pending',
        attempts        INTEGER NOT NULL DEFAULT 0,
        result          TEXT,
        error           TEXT,
        started_at      TEXT,
        finished_at     TEXT,
        PRIMARY KEY (job_id, issue_number)
    )
    """,
]


def init_db(db_path: Path | str | None = None) -> Path:
    """Create all tables if missing and reset interrupted item rows."""
    path = Path(db_path) if db_path is not None else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        for stmt in SCHEMA:
            conn.execute(stmt)
        # Crash recovery: anything left mid-run becomes pending again.
        conn.execute(
            "UPDATE items SET status='pending' WHERE status='running'"
        )
        conn.commit()

    return path


if __name__ == "__main__":
    created = init_db()
    print(f"Initialized schema at {created}")
