"""
ReuseLedger -- SQLite-backed bipartite reuse graph.

Maps ``(config_prefix, question_id)`` to physical WTB checkpoint entries.
This is the *physical* backing store for AG-UCT's ``materialized_keys``
(the paper's Path_t).

Schema::

    CREATE TABLE materialized_entries (
        prefix_key   TEXT NOT NULL,   -- JSON-encoded prefix tuple
        question_id  TEXT NOT NULL,
        execution_id TEXT NOT NULL,
        checkpoint_id TEXT NOT NULL,
        checkpoint_step INTEGER NOT NULL,
        checkpoint_db_path TEXT DEFAULT '',
        state_hash   TEXT DEFAULT '',
        created_at   TEXT NOT NULL,
        PRIMARY KEY (prefix_key, question_id)
    );
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple

from .config_types import RAGConfig

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS materialized_entries (
    prefix_key        TEXT    NOT NULL,
    question_id       TEXT    NOT NULL,
    execution_id      TEXT    NOT NULL,
    checkpoint_id     TEXT    NOT NULL,
    checkpoint_step   INTEGER NOT NULL DEFAULT 0,
    checkpoint_db_path TEXT   DEFAULT '',
    state_hash        TEXT    DEFAULT '',
    created_at        TEXT    NOT NULL,
    PRIMARY KEY (prefix_key, question_id)
);
"""

_INSERT_SQL = """
INSERT OR REPLACE INTO materialized_entries
    (prefix_key, question_id, execution_id, checkpoint_id,
     checkpoint_step, checkpoint_db_path, state_hash, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_SQL = """
SELECT execution_id, checkpoint_id, checkpoint_step,
       checkpoint_db_path, state_hash, created_at
FROM materialized_entries
WHERE prefix_key = ? AND question_id = ?;
"""

_COUNT_SQL = "SELECT COUNT(*) FROM materialized_entries;"

_ALL_KEYS_SQL = """
SELECT prefix_key, question_id FROM materialized_entries;
"""


def _encode_prefix(prefix: Tuple[str, ...]) -> str:
    return json.dumps(prefix, ensure_ascii=True)


def _decode_prefix(raw: str) -> Tuple[str, ...]:
    return tuple(json.loads(raw))


@dataclass
class MaterializedEntry:
    """A single entry in the reuse ledger.

    Represents a materialized ``(prefix, question_id)`` pair backed by a
    concrete WTB execution + checkpoint.
    """

    prefix: Tuple[str, ...]
    question_id: str
    execution_id: str
    checkpoint_id: str
    checkpoint_step: int = 0
    checkpoint_db_path: str = ""
    state_hash: str = ""
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ReuseLedger:
    """Bipartite reuse graph backed by a SQLite database.

    Thread-safe: each public method acquires an internal lock.

    Parameters
    ----------
    db_path
        Path to the SQLite database file.  Use ``":memory:"`` for an
        ephemeral in-memory ledger (useful for testing).
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level="DEFERRED",
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(_CREATE_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record(
        self,
        prefix: Tuple[str, ...],
        question_id: str,
        execution_id: str,
        checkpoint_id: str,
        checkpoint_step: int = 0,
        checkpoint_db_path: str = "",
        state_hash: str = "",
    ) -> MaterializedEntry:
        """Insert or update a materialized entry."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                _INSERT_SQL,
                (
                    _encode_prefix(prefix),
                    question_id,
                    execution_id,
                    checkpoint_id,
                    checkpoint_step,
                    checkpoint_db_path,
                    state_hash,
                    now,
                ),
            )
            self._conn.commit()
        return MaterializedEntry(
            prefix=prefix,
            question_id=question_id,
            execution_id=execution_id,
            checkpoint_id=checkpoint_id,
            checkpoint_step=checkpoint_step,
            checkpoint_db_path=checkpoint_db_path,
            state_hash=state_hash,
        )

    def lookup(
        self,
        prefix: Tuple[str, ...],
        question_id: str,
    ) -> Optional[MaterializedEntry]:
        """Return the entry for an exact ``(prefix, question_id)`` pair."""
        with self._lock:
            row = self._conn.execute(
                _SELECT_SQL, (_encode_prefix(prefix), question_id)
            ).fetchone()
        if row is None:
            return None
        return MaterializedEntry(
            prefix=prefix,
            question_id=question_id,
            execution_id=row[0],
            checkpoint_id=row[1],
            checkpoint_step=row[2],
            checkpoint_db_path=row[3],
            state_hash=row[4],
            created_at=datetime.fromisoformat(row[5]),
        )

    def longest_matching_prefix(
        self,
        config: RAGConfig,
        question_id: str,
    ) -> Tuple[int, Optional[MaterializedEntry]]:
        """Find the deepest cached prefix for ``(config, question_id)``.

        Searches from ``depth=5`` (full config) down to ``depth=1``.
        Returns ``(depth, entry)`` where *depth* is the number of leading
        slots that matched.  ``(0, None)`` means a complete cache miss.
        """
        for depth in range(5, 0, -1):
            entry = self.lookup(config.prefix(depth), question_id)
            if entry is not None:
                return depth, entry
        return 0, None

    # ------------------------------------------------------------------
    # Bulk / introspection helpers
    # ------------------------------------------------------------------

    def materialized_keys(self) -> Set[Hashable]:
        """Return all ``(prefix_tuple, question_id)`` pairs as a set.

        Compatible with ``SearchContext.materialized_keys`` in AG-UCT.
        """
        with self._lock:
            rows = self._conn.execute(_ALL_KEYS_SQL).fetchall()
        return {(_decode_prefix(r[0]), r[1]) for r in rows}

    def count(self) -> int:
        with self._lock:
            return self._conn.execute(_COUNT_SQL).fetchone()[0]

    def record_all_prefixes(
        self,
        config: RAGConfig,
        question_id: str,
        execution_id: str,
        checkpoints: List[Dict[str, Any]],
        checkpoint_db_path: str = "",
    ) -> List[MaterializedEntry]:
        """Record entries for every prefix depth of *config* for one question.

        ``checkpoints`` should be ordered by step (ascending).  Each
        checkpoint is expected to have at least ``id`` (or ``checkpoint_id``)
        and ``step`` keys.

        Depth 1 corresponds to after the frame selection (no node executed),
        depths 2-5 correspond to after query/retrieval/reranking/generation.
        """
        entries: List[MaterializedEntry] = []
        for depth in range(1, 6):
            cp_idx = depth - 1
            if cp_idx < len(checkpoints):
                cp = checkpoints[cp_idx]
                cp_id = str(cp.get("checkpoint_id", cp.get("id", "")))
                step = cp.get("step", depth)
            else:
                cp_id = checkpoints[-1].get("checkpoint_id",
                                            checkpoints[-1].get("id", "")) if checkpoints else ""
                step = depth

            entry = self.record(
                prefix=config.prefix(depth),
                question_id=question_id,
                execution_id=execution_id,
                checkpoint_id=str(cp_id),
                checkpoint_step=step,
                checkpoint_db_path=checkpoint_db_path,
            )
            entries.append(entry)
        return entries

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "ReuseLedger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
