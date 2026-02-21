from __future__ import annotations
from prometheus_client import make_asgi_app # Importe isso
from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="PEDE Passos Mágicos - Defasagem Risk API", version="1.0.0")

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(router)
