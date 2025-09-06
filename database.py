"""
Database helper module.

This module centralises access to the SQLite database.  It lazily opens a
singleton connection and provides helper functions for executing queries and
returning results as dictionaries.  The database path defaults to the
`project_data/erp.db` in the repository root, but you can override it by
setting the `ERP_DB_PATH` environment variable.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterable, List, Optional, Tuple, Union

_connection: Optional[sqlite3.Connection] = None
_lock = threading.Lock()


def get_db_path() -> str:
    """Return the path to the SQLite database.

    The path can be overridden via the `ERP_DB_PATH` environment variable.  If
    not set, it defaults to `project_data/erp.db` relative to this file.
    """
    env_path = os.environ.get("ERP_DB_PATH")
    if env_path:
        return env_path
    # Default relative to project root (database.py is in project root)
    return os.path.join(os.path.dirname(__file__), 'project_data', 'erp.db')


def get_connection() -> sqlite3.Connection:
    """Get or create a global SQLite connection.

    SQLite connections are not thread‑safe by default.  We reuse a single
    connection with `check_same_thread=False` and protect all operations with
    a lock.  Each row is returned as a `sqlite3.Row` object so that columns
    can be accessed by name or index.
    """
    global _connection
    if _connection is None:
        with _lock:
            if _connection is None:
                db_path = get_db_path()
                conn = sqlite3.connect(db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                _connection = conn
    return _connection


@contextmanager
def transaction() -> Iterable[sqlite3.Cursor]:
    """Context manager for executing a series of SQL statements in a transaction.

    Usage:

    ```python
    with database.transaction() as cur:
        cur.execute("INSERT INTO customers (name) VALUES (?)", ("Alice",))
        # ... more statements ...
    ```
    
    Any exception raised inside the block will roll back the transaction.
    Otherwise the transaction is committed when the block exits.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def execute(sql: str, params: Union[Tuple[Any, ...], List[Any], None] = None) -> sqlite3.Cursor:
    """Execute a single SQL statement and return the cursor.

    This helper obtains the global connection, executes the SQL with the
    provided parameters, commits the transaction and returns the cursor.  It
    should be used for non‑select statements.  If you need to run multiple
    statements in one transaction, use the :func:`transaction` context manager
    instead.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(sql, params or [])
        conn.commit()
        return cur
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def query(sql: str, params: Union[Tuple[Any, ...], List[Any], None] = None) -> List[sqlite3.Row]:
    """Execute a SELECT statement and return a list of rows.

    Each row is a `sqlite3.Row` object, supporting both numeric and named
    indexing.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(sql, params or [])
        rows = cur.fetchall()
        return rows
    finally:
        cur.close()


def query_one(sql: str, params: Union[Tuple[Any, ...], List[Any], None] = None) -> Optional[sqlite3.Row]:
    """Execute a SELECT statement and return a single row or None.
    """
    rows = query(sql, params)
    return rows[0] if rows else None


def to_dicts(rows: Iterable[sqlite3.Row]) -> List[dict]:
    """Convert an iterable of sqlite3.Row objects to a list of dicts.
    """
    return [dict(row) for row in rows]
