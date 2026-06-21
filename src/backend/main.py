"""FastAPI backend for the deepfake-detection app.

Endpoints (all HTTP; the Streamlit frontend polls /status):
  GET  /health     - liveness
  GET  /pipelines  - static metadata for every pipeline (+ availability)
  GET  /status     - resident pipeline, per-pipeline state, device, busy flag
  POST /select     - clear GPU + warm the requested pipeline (background)
  POST /predict    - run the warm pipeline on an uploaded image, persist result

Run: uvicorn src.backend.main:app --port 8000
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from . import config, pipelines
from .manager import ResidencyManager
from .schemas import (
    ExplainResponse,
    PredictionResponse,
    SelectRequest,
    SelectResponse,
    StatusResponse,
)

manager = ResidencyManager()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[backend] inference device = {manager.device}  "
          f"(cuda available: {torch.cuda.is_available()}; DF_DEVICE={os.environ.get('DF_DEVICE')!r})")
    yield


app = FastAPI(title="Deepfake Detection — pipeline server", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(manager.device)}


@app.get("/pipelines")
def get_pipelines():
    return pipelines.list_pipelines()


@app.get("/status", response_model=StatusResponse)
def get_status():
    return manager.status()


@app.post("/select", response_model=SelectResponse)
def select(req: SelectRequest):
    try:
        return manager.select(req.key)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown pipeline key: {req.key!r}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)):
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    filename = image.filename or "upload.png"
    try:
        return manager.predict(data, filename)
    except RuntimeError as e:
        # No warm pipeline yet — 409 Conflict so the UI can prompt a /select.
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:  # pragma: no cover - surfaced to the client
        return JSONResponse(status_code=500, content={"detail": f"inference failed: {e}"})


@app.post("/explain", response_model=ExplainResponse)
async def explain(image: UploadFile = File(...)):
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    filename = image.filename or "upload.png"
    try:
        return manager.explain(data, filename)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:  # pragma: no cover - surfaced to the client
        return JSONResponse(status_code=500, content={"detail": f"explanation failed: {e}"})
