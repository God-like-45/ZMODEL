"""Database configuration, environment loading, connection pooling, and credential masking."""

import os
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.exc import SQLAlchemyError


DEFAULT_POSTGRES_URL = "postgresql+psycopg://zmodel_user:zmodel_password@localhost:5432/zmodel_db"
DEFAULT_SQLITE_URL = "sqlite:///zmodel_experiments.db"


def load_dotenv_file(filepath: str = ".env") -> None:
    """Load key-value environment variables from local .env file if present."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


load_dotenv_file()


def get_database_url() -> str:
    """
    Retrieve database URL from environment or return default.
    Checks DATABASE_URL or DB_URL environment variables.
    If DB_ALLOW_SQLITE_FALLBACK is set and DATABASE_URL is not provided, uses SQLite.
    """
    url = os.environ.get("DATABASE_URL") or os.environ.get("DB_URL")
    if url and url.strip():
        return url.strip()

    allow_sqlite = os.environ.get("DB_ALLOW_SQLITE_FALLBACK", "false").lower() in ("true", "1", "yes")
    if allow_sqlite:
        sqlite_path = os.environ.get("SQLITE_DATABASE_PATH", "zmodel_experiments.db")
        return f"sqlite:///{sqlite_path}"

    return DEFAULT_POSTGRES_URL


def mask_database_url(url: Optional[str] = None) -> str:
    """
    Mask password in database URL for safe logging and UI display.
    Example: postgresql+psycopg://user:password@host:5432/db -> postgresql+psycopg://user:***@host:5432/db
    """
    if url is not None and not str(url).strip():
        return "Not Configured"
    target = url or get_database_url()
    if not target:
        return "Not Configured"
    
    # Check for credentials in URL (protocol://user:password@host...)
    pattern = r"://([^:]+):([^@]+)@"
    if re.search(pattern, target):
        return re.sub(pattern, r"://\1:***@", target)
    return target


def get_connection_pool_kwargs(url: str) -> Dict[str, Any]:
    """Return appropriate connection pool keyword arguments for engine creation."""
    kwargs: Dict[str, Any] = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # PostgreSQL pool settings
        pool_size = int(os.environ.get("DB_POOL_SIZE", "5"))
        max_overflow = int(os.environ.get("DB_MAX_OVERFLOW", "10"))
        pool_timeout = int(os.environ.get("DB_POOL_TIMEOUT_SECONDS", "10"))
        pool_recycle = int(os.environ.get("DB_POOL_RECYCLE_SECONDS", "1800"))
        connect_timeout = int(os.environ.get("DB_CONNECT_TIMEOUT", "1"))
        kwargs.update({
            "pool_size": pool_size,
            "max_overflow": max_overflow,
            "pool_timeout": pool_timeout,
            "pool_recycle": pool_recycle,
            "pool_pre_ping": True,
            "connect_args": {"connect_timeout": connect_timeout},
        })
    return kwargs


_HEALTH_CACHE: Optional[Dict[str, Any]] = None
_HEALTH_CACHE_TIME: float = 0.0
_HEALTH_CACHE_URL: Optional[str] = None


def check_database_connection(
    engine: Optional[Engine] = None,
    timeout_seconds: int = 1,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Execute a low-overhead health check against the configured database.
    Does NOT throw unhandled exceptions; returns a structured status dictionary.
    Caches results for 5 seconds when engine is not explicitly supplied.
    """
    global _HEALTH_CACHE, _HEALTH_CACHE_TIME, _HEALTH_CACHE_URL
    import time

    target_url = get_database_url()
    now = time.time()

    if not force_refresh and engine is None and _HEALTH_CACHE is not None:
        if _HEALTH_CACHE_URL == target_url and (now - _HEALTH_CACHE_TIME) < 5.0:
            return _HEALTH_CACHE

    created_engine = False
    test_engine = engine
    dialect = "unknown"

    try:
        if test_engine is None:
            # Create a short-lived test engine with quick connect timeout
            connect_kwargs = get_connection_pool_kwargs(target_url)
            if not target_url.startswith("sqlite"):
                # Pass short connect timeout to driver
                connect_kwargs.setdefault("connect_args", {})["connect_timeout"] = timeout_seconds
            test_engine = create_engine(target_url, **connect_kwargs)
            created_engine = True

        dialect = test_engine.dialect.name

        with test_engine.connect() as conn:
            # Simple ping
            conn.execute(text("SELECT 1"))
            
            # Inspect table presence
            from sqlalchemy import inspect
            inspector = inspect(conn)
            table_names = inspector.get_table_names()

        res = {
            "connected": True,
            "dialect": dialect,
            "database_url_masked": mask_database_url(target_url),
            "table_count": len(table_names),
            "tables": table_names,
            "error": None,
        }
        if engine is None:
            _HEALTH_CACHE = res
            _HEALTH_CACHE_TIME = now
            _HEALTH_CACHE_URL = target_url
        return res
    except SQLAlchemyError as err:
        # Sanitized error message without leaking raw passwords
        err_str = str(err)
        # Redact any password from error string if present
        err_sanitized = re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", err_str)
        res = {
            "connected": False,
            "dialect": dialect,
            "database_url_masked": mask_database_url(target_url),
            "table_count": 0,
            "tables": [],
            "error": err_sanitized,
        }
        if engine is None:
            _HEALTH_CACHE = res
            _HEALTH_CACHE_TIME = now
            _HEALTH_CACHE_URL = target_url
        return res
    except Exception as ex:
        res = {
            "connected": False,
            "dialect": dialect,
            "database_url_masked": mask_database_url(target_url),
            "table_count": 0,
            "tables": [],
            "error": str(ex),
        }
        if engine is None:
            _HEALTH_CACHE = res
            _HEALTH_CACHE_TIME = now
            _HEALTH_CACHE_URL = target_url
        return res
    finally:
        if created_engine and test_engine is not None:
            test_engine.dispose()
