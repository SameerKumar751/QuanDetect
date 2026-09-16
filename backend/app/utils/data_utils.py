"""
Lightweight session store.

Each dataset upload gets a session_id. We keep the raw dataframe, the
processed train/test splits, fitted preprocessing objects and trained
models all keyed by session_id so multiple users / uploads can be
handled concurrently without clobbering each other. For a production
deployment this would be swapped for Redis / a database, but an
in-process dict + joblib disk backup is sufficient for a demo platform
and keeps the module dependency-free.
"""
import os
import uuid
import joblib
from typing import Any, Dict, Optional

from app.config import SESSION_STORE_DIR

# In-memory cache. Falls back to disk (joblib) so the server can restart
# without losing sessions during development.
_SESSIONS: Dict[str, Dict[str, Any]] = {}


def new_session() -> str:
    session_id = str(uuid.uuid4())
    _SESSIONS[session_id] = {}
    return session_id


def _session_path(session_id: str) -> str:
    return os.path.join(SESSION_STORE_DIR, f"{session_id}.joblib")


def save_session(session_id: str, data: Dict[str, Any]) -> None:
    """Merge `data` into the existing session and persist to disk."""
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = {}
    _SESSIONS[session_id].update(data)
    try:
        joblib.dump(_SESSIONS[session_id], _session_path(session_id))
    except Exception:
        # Disk persistence is a best-effort convenience; in-memory cache
        # is the source of truth during the life of the process.
        pass


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    if session_id in _SESSIONS:
        return _SESSIONS[session_id]
    path = _session_path(session_id)
    if os.path.exists(path):
        data = joblib.load(path)
        _SESSIONS[session_id] = data
        return data
    return None


def require_session(session_id: str) -> Dict[str, Any]:
    data = get_session(session_id)
    if data is None:
        raise KeyError(f"Session '{session_id}' not found. Upload a dataset first.")
    return data
