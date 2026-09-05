import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache

from sqlalchemy import DateTime, ForeignKey, String, Text, create_engine, func, select
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session as SASession,
    mapped_column,
    relationship,
    sessionmaker,
)

from app.core.config import get_settings


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[str] = mapped_column(String(200), default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.id"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id"))
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    sources_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    session: Mapped["ChatSession"] = relationship(back_populates="messages")


@dataclass
class Message:
    """Plain-data view of a ChatMessage, decoupled from the ORM object
    so callers (e.g. app.generation.chain) don't need a live session."""
    role: str
    content: str
    sources: list = field(default_factory=list)
    created_at: datetime | None = None


@lru_cache
def _engine():
    settings = get_settings()
    # check_same_thread=False: FastAPI's sync route handlers run in a
    # threadpool, so different requests may use this engine from
    # different threads. Each call below still opens/closes its own
    # short-lived Session — no state is shared across requests.
    return create_engine(
        f"sqlite:///{settings.sqlite_db_path}",
        connect_args={"check_same_thread": False},
    )


@lru_cache
def _session_factory() -> sessionmaker:
    return sessionmaker(bind=_engine(), expire_on_commit=False)


@contextmanager
def _session():
    db: SASession = _session_factory()()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create tables if they don't exist yet. Safe to call on every
    startup — create_all is idempotent."""
    Base.metadata.create_all(_engine())


def create_session(title: str = "New chat") -> str:
    with _session() as db:
        chat_session = ChatSession(title=title)
        db.add(chat_session)
        db.flush()  # populate chat_session.id (the default lambda) before we read it
        return chat_session.id


def session_exists(session_id: str) -> bool:
    with _session() as db:
        return db.get(ChatSession, session_id) is not None


def list_sessions() -> list[dict]:
    """Most recently created first, with each session's message count."""
    with _session() as db:
        rows = db.execute(
            select(
                ChatSession.id,
                ChatSession.title,
                ChatSession.created_at,
                func.count(ChatMessage.id).label("message_count"),
            )
            .outerjoin(ChatMessage, ChatMessage.session_id == ChatSession.id)
            .group_by(ChatSession.id)
            .order_by(ChatSession.created_at.desc())
        ).all()
        return [
            {
                "id": r.id,
                "title": r.title,
                "created_at": r.created_at.isoformat(),
                "message_count": r.message_count,
            }
            for r in rows
        ]


def rename_session(session_id: str, title: str) -> None:
    with _session() as db:
        chat_session = db.get(ChatSession, session_id)
        if chat_session:
            chat_session.title = title


def delete_session(session_id: str) -> None:
    with _session() as db:
        chat_session = db.get(ChatSession, session_id)
        if chat_session:
            db.delete(chat_session)  # cascades to messages, see relationship above
        # note: _session()'s db.commit() on exit is what actually persists
        # this — forgetting that commit is exactly what makes a delete a
        # silent no-op, so it's worth keeping this comment as a tripwire.


def add_message(session_id: str, role: str, content: str, sources: list | None = None) -> None:
    with _session() as db:
        # Check BEFORE inserting the new row, so this doesn't depend on
        # SQLAlchemy's autoflush timing (a select() would otherwise flush
        # a pending add() first and silently shift the count by one).
        is_first_user_message = role == "user" and db.scalar(
            select(func.count(ChatMessage.id)).where(
                ChatMessage.session_id == session_id, ChatMessage.role == "user"
            )
        ) == 0

        db.add(
            ChatMessage(
                session_id=session_id,
                role=role,
                content=content,
                sources_json=json.dumps(sources or []),
            )
        )
        # First user message becomes the session title (trimmed), so the
        # sidebar shows something more useful than "New chat".
        if is_first_user_message:
            chat_session = db.get(ChatSession, session_id)
            if chat_session:
                chat_session.title = content.strip().replace("\n", " ")[:60] or "New chat"


def get_messages(session_id: str, limit: int | None = None) -> list[Message]:
    with _session() as db:
        rows = db.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.asc())
        ).all()
        messages = [
            Message(
                role=r.role,
                content=r.content,
                sources=json.loads(r.sources_json or "[]"),
                created_at=r.created_at,
            )
            for r in rows
        ]
    if limit is not None and len(messages) > limit:
        messages = messages[-limit:]
    return messages