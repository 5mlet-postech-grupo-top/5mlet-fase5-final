from __future__ import annotations

from fastapi import FastAPI
from app.routes import router


def create_app() -> FastAPI:
    """PEDE Passos Mágicos - Defasagem Risk API."""
    app = FastAPI(title="PEDE Passos Mágicos - Defasagem Risk API", version="1.0.0")
    app.include_router(router)
    return app


# instantiate the default app for uvicorn / normal imports
app = create_app()
