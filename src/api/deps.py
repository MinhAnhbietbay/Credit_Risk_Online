from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from src.online.engine import get_session

__all__ = ["get_db"]

def get_db() -> Iterator[Session]:
    session = get_session()
    try:
        yield session
    finally:
        session.close()
