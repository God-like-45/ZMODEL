"""Database engine management, session factory, context managers, and schema initialization."""

from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from zmodel.db.config import get_connection_pool_kwargs, get_database_url
from zmodel.db.models import Base

_ENGINE: Optional[Engine] = None
_SESSION_FACTORY: Optional[sessionmaker] = None


def get_engine(url: Optional[str] = None) -> Engine:
    """
    Get or create singleton SQLAlchemy Engine.
    Uses configured pooling and pre-ping connectivity tests.
    """
    global _ENGINE, _SESSION_FACTORY
    target_url = url or get_database_url()
    
    # If engine exists with different URL, dispose and recreate
    if _ENGINE is not None:
        if str(_ENGINE.url) == target_url:
            return _ENGINE
        _ENGINE.dispose()
        _ENGINE = None
        _SESSION_FACTORY = None

    pool_kwargs = get_connection_pool_kwargs(target_url)
    _ENGINE = create_engine(target_url, **pool_kwargs)
    _SESSION_FACTORY = sessionmaker(bind=_ENGINE, expire_on_commit=False)
    return _ENGINE


def reset_engine() -> None:
    """Dispose active engine and reset session factory singleton."""
    global _ENGINE, _SESSION_FACTORY
    if _ENGINE is not None:
        _ENGINE.dispose()
        _ENGINE = None
        _SESSION_FACTORY = None


def get_session_factory(engine: Optional[Engine] = None) -> sessionmaker:
    """Return sessionmaker bound to active or provided engine."""
    global _SESSION_FACTORY
    if engine is not None:
        return sessionmaker(bind=engine, expire_on_commit=False)
    if _SESSION_FACTORY is None:
        get_engine()
    return _SESSION_FACTORY


@contextmanager
def session_scope(engine: Optional[Engine] = None) -> Generator[Session, None, None]:
    """
    Context manager providing a transactional database session scope.
    Commits on successful completion, rolls back on exception, and always closes.
    Transactions are short-lived and never held open across long-running benchmark runs.
    """
    factory = get_session_factory(engine)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Optional[Engine] = None) -> None:
    """
    Safely create database tables if they do not exist.
    Does not drop or truncate existing data.
    """
    target_engine = engine or get_engine()
    Base.metadata.create_all(bind=target_engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding a transactional database session scope.
    Closes and commits or rolls back session automatically.
    """
    with session_scope() as session:
        yield session

