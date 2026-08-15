"""Database interface for PV-SQL.

PV-SQL only ever touches a database through the five methods on `DatabaseEnv`.
Implement those against any engine and the rest of the method works unchanged;
`SQLiteEnv` below is the reference implementation.

The probing loop issues model-authored SELECTs against a live database, so the
connection you hand it should be **read-only and disposable**. See the
"Security" section of the README before pointing this at anything real.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# (column_names, rows, error_message)
QueryResult = Tuple[List[str], List[tuple], Optional[str]]


class DatabaseEnv(ABC):
    """The contract PV-SQL depends on."""

    @abstractmethod
    def execute(self, sql: str) -> QueryResult:
        """Run `sql`. Return (columns, rows, error). Never raise -- report the
        error string instead, since the repair loop feeds it back to the model."""

    @abstractmethod
    def explain_err(self, sql: str) -> Optional[str]:
        """Validate `sql` without running it. Return None if valid, else the
        error message."""

    @abstractmethod
    def schema_overview(self) -> str:
        """A compact text rendering of tables, columns and foreign keys."""

    @abstractmethod
    def get_foreign_keys(self) -> Dict[str, List[Dict[str, str]]]:
        """Map table -> list of {from_table, from_column, to_table, to_column}."""

    @abstractmethod
    def sample_values(self, table: str, column: str, limit: int = 3) -> List[Any]:
        """A few distinct non-null values, used to pin down literal formats."""


class SQLiteEnv(DatabaseEnv):
    """SQLite-backed environment. Point it at any .sqlite/.db file.

    >>> env = SQLiteEnv("examples/university.sqlite")
    >>> env.schema_overview()[:40]
    'Table student: ...'
    """

    def __init__(self, db_path: str | Path, read_only: bool = True, timeout: float = 30.0):
        self.db_path = str(db_path)
        self.read_only = read_only
        self.timeout = timeout
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
        self._fk_cache: Optional[Dict[str, List[Dict[str, str]]]] = None

    # -- dataset conveniences ------------------------------------------------

    @classmethod
    def from_bird(cls, db_root: str | Path, db_id: str, **kw) -> "SQLiteEnv":
        """BIRD layout: <db_root>/<db_id>/<db_id>.sqlite"""
        return cls(Path(db_root) / db_id / f"{db_id}.sqlite", **kw)

    @classmethod
    def from_spider(cls, db_root: str | Path, db_id: str, **kw) -> "SQLiteEnv":
        """Spider layout: <db_root>/<db_id>/<db_id>.sqlite"""
        return cls(Path(db_root) / db_id / f"{db_id}.sqlite", **kw)

    # -- connection ----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            uri = f"file:{Path(self.db_path).as_posix()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=self.timeout)
        else:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        # Benchmark databases carry mixed and sometimes invalid encodings.
        conn.text_factory = lambda b: b.decode(errors="ignore")
        return conn

    # -- DatabaseEnv contract ------------------------------------------------

    def execute(self, sql: str) -> QueryResult:
        conn = None
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
            return cols, rows, None
        except Exception as e:
            return [], [], str(e)
        finally:
            if conn is not None:
                conn.close()

    def explain_err(self, sql: str) -> Optional[str]:
        conn = None
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(f"EXPLAIN QUERY PLAN {sql.strip().rstrip(';')}")
            cur.fetchall()
            return None
        except Exception as e:
            return str(e)
        finally:
            if conn is not None:
                conn.close()

    def list_tables(self) -> List[str]:
        _, rows, _ = self.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        return [r[0] for r in rows] if rows else []

    def schema_overview(self) -> str:
        out: List[str] = []
        for t in self.list_tables():
            _, info, _ = self.execute(f"PRAGMA table_info(`{t}`);")
            col_names = [r[1] for r in info] if info else []
            _, fk_rows, _ = self.execute(f"PRAGMA foreign_key_list(`{t}`);")
            fks = [f"{r[3]} -> {r[2]}.{r[4]}" for r in fk_rows] if fk_rows else []
            line = f"Table {t}: {col_names}"
            if fks:
                line += f" | FKs: {fks}"
            out.append(line)
        return "\n".join(out)

    def get_foreign_keys(self) -> Dict[str, List[Dict[str, str]]]:
        if self._fk_cache is None:
            cache: Dict[str, List[Dict[str, str]]] = {}
            for t in self.list_tables():
                _, fk_rows, _ = self.execute(f"PRAGMA foreign_key_list(`{t}`);")
                fks = [
                    {
                        "from_table": t,
                        "from_column": r[3],
                        "to_table": r[2],
                        "to_column": r[4],
                    }
                    for r in (fk_rows or [])
                ]
                if fks:
                    cache[t] = fks
            self._fk_cache = cache
        return self._fk_cache

    def sample_values(self, table: str, column: str, limit: int = 3) -> List[Any]:
        _, rows, err = self.execute(
            f"SELECT DISTINCT `{column}` FROM `{table}` "
            f"WHERE `{column}` IS NOT NULL LIMIT {int(limit)}"
        )
        if err or not rows:
            return []
        vals = []
        for r in rows:
            v = r[0]
            if isinstance(v, str) and len(v) > 60:
                v = v[:60] + "..."
            vals.append(v)
        return vals
