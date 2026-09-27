from __future__ import annotations

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from server.core import config
from server.core import model as ml
from server.core import storage
from server.core.logging_utils import JsonLogger, get_logger

SERVICE_NAME = "galaxeye-tile-classifier"
SERVICE_VERSION = "0.1.0"


class ClassifyResponse(BaseModel):
    prediction_id: str
    tile_name: str
    tile_sha256: str
    label: str
    confidence: float
    needs_review: bool
    review_threshold: float
    probabilities: dict[str, float]
    model_version: str | None
    latency_ms: float
    created_at: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model: dict[str, Any]
    evaluation: dict[str, Any] | None
    stored_predictions: int


class PredictionRow(BaseModel):
    prediction_id: str
    tile_name: str
    tile_sha256: str
    label: str
    confidence: float
    needs_review: bool
    model_version: str
    latency_ms: float
    created_at: str


class PredictionListResponse(BaseModel):
    count: int
    returned: int
    predictions: list[PredictionRow]


def _load_eval_report() -> dict[str, Any] | None:
    if not config.EVAL_REPORT_PATH.is_file():
        return None
    try:
        report = json.loads(config.EVAL_REPORT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    slim: dict[str, Any] = {
        "accuracy": report.get("accuracy"),
        "macro_f1": report.get("macro_f1"),
        "mean_confidence": report.get("mean_confidence"),
        "flagged_fraction": report.get("flagged_fraction"),
        "review_threshold": report.get("review_threshold"),
        "per_class": report.get("per_class"),
        "confusion_matrix": report.get("confusion_matrix"),
        "top_confusions": report.get("top_confusions"),
        "split": report.get("split"),
    }
    return slim


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log = get_logger("service", filename="service.log")
    boot = time.perf_counter()
    log.info("service_start", pid=os.getpid(), device=config.DEVICE)
    try:
        app.state.model = ml.load_classifier()
    except FileNotFoundError as exc:
        log.error("model_load_failed", reason=str(exc))
        raise
    app.state.model_loaded_seconds = round(time.perf_counter() - boot, 3)
    app.state.started_at = time.time()
    app.state.log = log
    log.info(
        "model_ready",
        model_version=app.state.model.model_version,
        load_seconds=app.state.model_loaded_seconds,
    )
    try:
        yield
    finally:
        log.info("service_stop")
        log.close()


app = FastAPI(
    title="GalaxEye Tile Classifier",
    version=SERVICE_VERSION,
    summary="Offline land-use classification for satellite image tiles.",
    lifespan=lifespan,
)


def get_model() -> ml.TileClassifier:
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503, detail="model is not loaded; the service is still starting"
        )
    return model


def get_log() -> JsonLogger:
    return getattr(app.state, "log", None) or get_logger("service", filename="service.log", echo=False)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        uptime_seconds=round(time.time() - getattr(app.state, "started_at", time.time()), 3),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
def model_info(model: ml.TileClassifier = Depends(get_model)) -> ModelInfoResponse:
    return ModelInfoResponse(
        model=model.info(),
        evaluation=_load_eval_report(),
        stored_predictions=storage.count_predictions(),
    )


@app.post("/classify", response_model=ClassifyResponse, tags=["inference"])
async def classify(
    file: UploadFile = File(..., description="A satellite image tile (PNG/JPEG)."),
    review_threshold: float = Form(
        default=config.REVIEW_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Below this softmax confidence the tile is flagged for human review.",
    ),
    model: ml.TileClassifier = Depends(get_model),
) -> ClassifyResponse:
    log = get_log()
    if file.size is not None and file.size > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"upload is {file.size} bytes, limit is {config.MAX_UPLOAD_BYTES} bytes"
            ),
        )
    if (
        file.content_type
        and file.content_type.lower() not in config.ALLOWED_CONTENT_TYPES
    ):
        raise HTTPException(
            status_code=415,
            detail=f"unsupported content type {file.content_type!r}",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    digest = storage.sha256_bytes(payload)

    started = time.perf_counter()
    try:
        result = await run_in_threadpool(
            model.predict_bytes, payload, review_threshold, config.DEVICE
        )
    except Exception as exc:
        log.warning(
            "classify_rejected",
            tile_name=file.filename,
            tile_sha256=digest[:12],
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=400, detail=f"could not decode the upload as an image: {exc}"
        ) from exc
    latency_ms = (time.perf_counter() - started) * 1000.0

    record = {
        "tile_name": file.filename or digest[:12],
        "tile_sha256": digest,
        "label": result["label"],
        "confidence": result["confidence"],
        "needs_review": result["needs_review"],
        "model_version": model.model_version or "unknown",
        "latency_ms": round(latency_ms, 2),
    }
    prediction_id = await run_in_threadpool(storage.append_prediction, record)
    created_at = storage.utc_now()

    log.info(
        "classified",
        prediction_id=prediction_id,
        tile_name=record["tile_name"],
        tile_sha256=digest[:12],
        label=result["label"],
        confidence=result["confidence"],
        needs_review=result["needs_review"],
        model_version=record["model_version"],
        latency_ms=record["latency_ms"],
    )

    return ClassifyResponse(
        prediction_id=prediction_id,
        review_threshold=review_threshold,
        model_version=model.model_version,
        created_at=created_at,
        latency_ms=record["latency_ms"],
        **result,
        tile_name=record["tile_name"],
        tile_sha256=digest,
    )


@app.get("/predictions", response_model=PredictionListResponse, tags=["inference"])
def list_predictions(
    label: str | None = Query(default=None, description="Exact label match."),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    max_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    needs_review: bool | None = Query(default=None),
    since: str | None = Query(default=None, description="ISO-8601 lower bound on created_at."),
    until: str | None = Query(default=None, description="ISO-8601 upper bound on created_at."),
    limit: int = Query(default=100, ge=1, le=10_000),
) -> PredictionListResponse:
    if label is not None and label not in config.CLASSES:
        raise HTTPException(
            status_code=422,
            detail=f"unknown label {label!r}; expected one of {config.CLASSES}",
        )
    rows = storage.query_predictions(
        label=label,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        needs_review=needs_review,
        since=since,
        until=until,
        limit=limit,
    )
    total = storage.count_predictions()
    return PredictionListResponse(
        count=total,
        returned=len(rows),
        predictions=[PredictionRow(**_coerce_row(row)) for row in rows],
    )


def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    def as_float(key: str) -> float:
        try:
            return float(row.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    return {
        "prediction_id": str(row.get("prediction_id", "")),
        "tile_name": str(row.get("tile_name", "")),
        "tile_sha256": str(row.get("tile_sha256", "")),
        "label": str(row.get("label", "")),
        "confidence": as_float("confidence"),
        "needs_review": str(row.get("needs_review", "")).strip().lower()
        in {"1", "true", "yes", "y", "t"},
        "model_version": str(row.get("model_version", "")),
        "latency_ms": as_float("latency_ms"),
        "created_at": str(row.get("created_at", "")),
    }
