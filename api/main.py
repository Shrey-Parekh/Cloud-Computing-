"""HTTP API for the ARIMA workload predictor.

Run locally:  uvicorn api.main:app --reload --port 8000
Endpoints  :  GET /api/health, POST /api/run, POST /api/suggest

All modelling is done by the existing ``src`` modules; this file only turns
their results into JSON. If ``web/dist`` exists (the built React app), it is
served at ``/`` so one process can host the whole site.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.arima_model import (
    MODE_MULTI_STEP,
    MODE_ONE_STEP,
    SEASONAL,
    ArimaError,
    forecast_future,
    predict_test_period,
    split_train_test,
    suggest_order,
)
from src.data_loader import DEFAULT_DAYS, DataLoadError, get_workload
from src.evaluation import EvaluationError, evaluate
from src.preprocessing import PreprocessingError, to_series

DATA_TTL_SECONDS = 3600
DIST = Path(__file__).resolve().parent.parent / "web" / "dist"

# --------------------------------------------------------------------------- #
# Data (loaded once, refreshed at most hourly, like the Streamlit cache was)
# --------------------------------------------------------------------------- #
_data_lock = threading.Lock()
_data: dict = {}


def load_data(refresh: bool = False):
    """Return (dataframe, info, version). ``version`` changes when data changes."""
    with _data_lock:
        stale = time.time() - _data.get("loaded_at", 0) > DATA_TTL_SECONDS
        if refresh or stale or "df" not in _data:
            try:
                df, info = get_workload(days=DEFAULT_DAYS, force_refresh=refresh)
            except (DataLoadError, PreprocessingError) as exc:
                if "df" not in _data:
                    raise HTTPException(503, f"Could not load workload data: {exc}") from exc
            else:
                _data.update(df=df, info=info, loaded_at=time.time(),
                             version=str(df["timestamp"].iloc[-1]))
        return _data["df"], _data["info"], _data["version"]


def _ts(index) -> list[str]:
    return [t.strftime("%Y-%m-%dT%H:00") for t in index]


def _ints(values) -> list[int]:
    return [int(round(float(v))) for v in values]


# --------------------------------------------------------------------------- #
# Model runs (cached per settings + data version: fits take seconds)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=32)
def _run(version: str, order: tuple, train_ratio: float, mode: str, horizon: int) -> dict:
    df, info, _ = load_data()
    series = to_series(df)
    train, test = split_train_test(series, train_ratio)
    predicted, aic = predict_test_period(train, test, order, mode)
    forecast = forecast_future(series, order, horizon)
    return {
        "data": {
            "source": info["source"],
            "message": info["message"],
            "rows": len(series),
            "imputed": int(df["imputed"].sum()),
            "mean": float(series.mean()),
            "peak": float(series.max()),
        },
        "order": list(order),
        "seasonal": list(SEASONAL),
        "mode": mode,
        "train_ratio": train_ratio,
        "aic": aic,
        "history": {"t": _ts(series.index), "y": _ints(series)},
        "n_train": len(train),
        "predicted": _ints(predicted),
        "metrics": evaluate(test, predicted),
        "forecast": {
            "t": _ts(forecast.index),
            "predicted": _ints(forecast["predicted"]),
            "lower": _ints(forecast["lower"]),
            "upper": _ints(forecast["upper"]),
        },
    }


class RunRequest(BaseModel):
    p: int = Field(2, ge=0, le=10)
    d: int = Field(1, ge=0, le=2)
    q: int = Field(2, ge=0, le=10)
    train_ratio: float = Field(0.8, ge=0.6, le=0.9)
    mode: str = MODE_ONE_STEP
    horizon: int = Field(24, ge=1, le=72)
    refresh: bool = False


class SuggestRequest(BaseModel):
    train_ratio: float = Field(0.8, ge=0.6, le=0.9)


def run(req: RunRequest):
    if req.mode not in (MODE_ONE_STEP, MODE_MULTI_STEP):
        raise HTTPException(422, f"Unknown mode {req.mode!r}.")
    _, _, version = load_data(refresh=req.refresh)
    try:
        return _run(version, (req.p, req.d, req.q), round(req.train_ratio, 2),
                    req.mode, req.horizon)
    except (ArimaError, EvaluationError) as exc:
        raise HTTPException(422, str(exc)) from exc


@asynccontextmanager
async def lifespan(_app):
    # Fit the default model in the background so the first visitor waits less.
    def warm():
        try:
            run(RunRequest())
        except Exception:  # best effort; a real request will report the error
            pass
    threading.Thread(target=warm, daemon=True).start()
    yield


app = FastAPI(title="Cloud Workload Prediction API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health():
    return {"ok": True}


app.post("/api/run")(run)


@app.post("/api/suggest")
def suggest(req: SuggestRequest):
    df, _, _ = load_data()
    try:
        train, _ = split_train_test(to_series(df), req.train_ratio)
        best, table = suggest_order(train)
    except ArimaError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"order": list(best), "candidates": table.head(10).to_dict(orient="records")}


if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
