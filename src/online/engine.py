from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import DATABASE_URL
from src.online.schema import Base

__all__ = ["get_engine", "get_session", "init_db", "drop_all", "table_names", "ping", "DATABASE_URL"]

@lru_cache(maxsize=8)
def get_engine(url: str = DATABASE_URL) -> Engine:
    return create_engine(url, echo=False, pool_pre_ping=True, future=True)


def get_session(url: str = DATABASE_URL) -> Session:
    return sessionmaker(bind=get_engine(url), expire_on_commit=False)()

def init_db(url: str = DATABASE_URL) -> list[str]:
    Base.metadata.create_all(get_engine(url))
    return table_names(url)

def drop_all(url: str = DATABASE_URL) -> None:
    Base.metadata.drop_all(get_engine(url))

def table_names(url: str = DATABASE_URL) -> list[str]:
    return sorted(inspect(get_engine(url)).get_table_names())

def ping(url: str = DATABASE_URL) -> tuple[bool, str]:
    from sqlalchemy import text
    try:
        with get_engine(url).connect() as conn:
            return True, str(conn.execute(text("select version()")).scalar())
    except Exception as exc:                    
        return False, f"{type(exc).__name__}: {exc}"
