"""SQLModel tables for sessions and transcripts, plus the SQLite engine."""
from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlmodel import Field, Session as DBSession, SQLModel, create_engine


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Session(SQLModel, table=True):
    """A live signing session."""

    id: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=_utcnow)
    keypoints_only: bool = True  # privacy default


class Transcript(SQLModel, table=True):
    """A recognized gloss / translated-text record tied to a session."""

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="session.id", index=True)
    t_start: float
    t_end: float
    gloss: str = ""
    text: str = ""
    # CTC peak timestamps (JSON-encoded list of floats) for the UI timeline.
    peaks: str = "[]"


DATABASE_URL = os.environ.get("SIGN_RT_DB_URL", "sqlite:///sign_rt.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


def init_db() -> None:
    """Create tables if they do not exist."""
    SQLModel.metadata.create_all(engine)


def get_session() -> DBSession:
    """Open a new DB session (caller closes / uses as context manager)."""
    return DBSession(engine)
