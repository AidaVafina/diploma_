from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings
from app.core.logging import setup_logging


def create_app() -> FastAPI:
    setup_logging()
    settings.ensure_directories()

    application = FastAPI(
        title="Scientific Journal Scan Pipeline",
        description="PDF intake, OCR preprocessing, and Surya-based page layout analysis",
        version="0.2.0",
    )
    application.include_router(router)
    application.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    return application


app = create_app()
