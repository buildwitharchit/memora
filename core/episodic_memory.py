"""
Tier 4 -- Episodic Memory.

A structured, timestamped event log backed by SQLite. Each row records the
user's intent and the interaction outcome for a single exchange.

Persistence: SQLite database at <store_path>/episodic.db.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class EpisodicEntry(BaseModel):
    """A single episodic log record."""

    id: int | None = None
    timestamp: str
    user_id: str
    session_id: str
    intent: str
    outcome: str


class EpisodicMemory:
    """Tier 4: SQLite-backed structured log of intents and outcomes."""

    TABLE_NAME = "episodic_log"

    def __init__(self, store_path: str = "./store") -> None:
        db_path = str(Path(store_path) / "episodic.db")
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self) -> None:
        """Ensure the episodic_log table exists."""
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                intent TEXT NOT NULL,
                outcome TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def add_entry(
        self,
        user_id: str,
        session_id: str,
        intent: str,
        outcome: str,
    ) -> int:
        """Insert an episodic entry. Returns the new row ID."""
        ts = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            f"""
            INSERT INTO {self.TABLE_NAME} (timestamp, user_id, session_id, intent, outcome)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ts, user_id, session_id, intent, outcome),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_recent(self, session_id: str, limit: int = 5) -> list[EpisodicEntry]:
        """Most recent N entries for a session, newest first."""
        rows = self._conn.execute(
            f"""
            SELECT id, timestamp, user_id, session_id, intent, outcome
            FROM {self.TABLE_NAME}
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [EpisodicEntry(**dict(row)) for row in rows]

    def get_all(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        intent_pattern: str | None = None,
        limit: int = 100,
    ) -> list[EpisodicEntry]:
        """Filtered query for the Episodic Log Viewer tab.

        Filters are optional and combined with AND. intent_pattern uses SQL LIKE.
        """
        clauses: list[str] = []
        params: list = []

        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if intent_pattern:
            clauses.append("intent LIKE ?")
            params.append(f"%{intent_pattern}%")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        rows = self._conn.execute(
            f"""
            SELECT id, timestamp, user_id, session_id, intent, outcome
            FROM {self.TABLE_NAME}
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [EpisodicEntry(**dict(row)) for row in rows]

    def count(self) -> int:
        """Total number of episodic entries."""
        row = self._conn.execute(
            f"SELECT COUNT(*) FROM {self.TABLE_NAME}"
        ).fetchone()
        return row[0]

    def reset(self) -> None:
        """Delete all episodic entries."""
        self._conn.execute(f"DELETE FROM {self.TABLE_NAME}")
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
