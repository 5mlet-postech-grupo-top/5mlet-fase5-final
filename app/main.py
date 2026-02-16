from __future__ import annotations

from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="PEDE Passos Mágicos - Defasagem Risk API", version="1.0.0")
app.include_router(router)
