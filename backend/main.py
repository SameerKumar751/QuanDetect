"""
QuanDetect Backend Entrypoint
==============================
Hybrid Quantum-Classical ML Platform for Early Disease Detection
(SIH26139).

Run with:
    uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from imaging.api.routes import router as imaging_router

app = FastAPI(
    title="QuanDetect API",
    description=(
        "Hybrid Quantum-Classical Machine Learning Platform for Early "
        "Disease Detection. Provides tabular baseline models, a PennyLane VQC, "
        "and an independent medical imaging (CT) pipeline for lung cancer detection."
    ),
    version="1.0.0",
)

# Allow the Vite dev server (and any deployed frontend) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Existing Tabular Pipeline Router
app.include_router(router, prefix="/api")

# Mount Independent Medical Imaging (CT) Pipeline Router
app.include_router(imaging_router, prefix="/api/imaging")


@app.get("/")
async def root():
    return {
        "message": "QuanDetect API is running.",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}
