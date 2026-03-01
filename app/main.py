from __future__ import annotations
from fastapi import FastAPI
from prometheus_client import make_asgi_app
from app.routes import router


def create_app() -> FastAPI:
    """PEDE Passos Mágicos - Defasagem Risk API."""
    app = FastAPI(title="PEDE Passos Mágicos - Defasagem Risk API", version="1.0.0")

    app.include_router(router)

    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    return app

app = create_app()