"""ackend: `uvicorn src.api.main:app --reload --port 8000`.

Chuẩn bị trước khi chạy (giống bản Streamlit):
    docker compose -f infra/docker-compose.yml up -d
    python scripts/ph1_db_seed.py
    python scripts/ph1_web_assets.py
"""

from __future__ import annotations

import os

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError

from src.api.routers import catalog, insights, predict, predictions, retrain, segmentation, stress

__all__ = ["app", "create_app", "ROUTERS", "DB_DOWN_DETAIL"]

DB_DOWN_DETAIL = (
    "Không kết nối được PostgreSQL. Bật DB rồi tải lại trang: "
    "`docker compose -f infra/docker-compose.yml up -d` rồi `python scripts/ph1_db_seed.py`."
)

ROUTERS: list[APIRouter] = [catalog.router, predict.router, retrain.router, predictions.router,
                            segmentation.router, insights.router, stress.router]

async def _db_down(_: Request, __: OperationalError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": DB_DOWN_DETAIL})


async def _missing_file(_: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})

def create_app() -> FastAPI:
    app = FastAPI(title="PH1 — Rủi ro tín dụng Home Credit")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(OperationalError, _db_down)
    app.add_exception_handler(FileNotFoundError, _missing_file)
    for router in ROUTERS:
        app.include_router(router, prefix="/api")
    return app

app = create_app()
