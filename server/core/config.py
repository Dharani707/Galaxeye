from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASSES: list[str] = [
    "AnnualCrop",
    "Forest",
    "Highway",
    "Industrial",
    "Residential",
    "River",
    "SeaLake",
]
NUM_CLASSES: int = len(CLASSES)
CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}

IMAGE_SIZE: int = 224
NORM_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
NORM_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

BACKBONE: str = "resnet18"
UNFREEZE: str = "none"

AUG_COPIES: int = 3
VAL_FRACTION: float = 0.20
SEED: int = 42

HEAD_LR: float = 1e-3
HEAD_EPOCHS: int = 300
HEAD_PATIENCE: int = 30
BACKBONE_LR: float = 1e-4
FINETUNE_EPOCHS: int = 30
FINETUNE_PATIENCE: int = 8

BATCH_SIZE: int = 64
NUM_WORKERS: int = 0
DEVICE: str = "cpu"

REVIEW_THRESHOLD: float = 0.60
CONFIDENCE_BINS: int = 10

MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
        "image/bmp",
        "image/tiff",
        "application/octet-stream",
    }
)


def _resolve(env_name: str, default: Path) -> Path:
    raw = os.getenv(env_name)
    return Path(raw).expanduser().resolve() if raw else default


DATA_DIR: Path = _resolve("GALAXEY_DATA_DIR", PROJECT_ROOT / "data")
CANDIDATE_DIR: Path = _resolve("GALAXEY_CANDIDATE_DIR", DATA_DIR / "candidate_tiles")
EVAL_SET_DIR: Path = _resolve("GALAXEY_EVAL_SET_DIR", DATA_DIR / "eval_set")
EVAL_LABELS_CSV: Path = _resolve("GALAXEY_EVAL_LABELS_CSV", DATA_DIR / "eval_labels.csv")

MODEL_PATH: Path = _resolve("GALAXEY_MODEL_PATH", PROJECT_ROOT / "model.pt")
PREDICTIONS_PATH: Path = _resolve("GALAXEY_PREDICTIONS_PATH", PROJECT_ROOT / "predictions.csv")
METRICS_DIR: Path = _resolve("GALAXEY_METRICS_DIR", PROJECT_ROOT / "metrics")
LOG_DIR: Path = _resolve("GALAXEY_LOG_DIR", PROJECT_ROOT / "logs")

EVAL_REPORT_PATH: Path = METRICS_DIR / "eval_report.json"
CONFUSION_MATRIX_PATH: Path = METRICS_DIR / "confusion_matrix.csv"
TRAIN_LOG_PATH: Path = LOG_DIR / "train_run.jsonl"

PREDICTION_COLUMNS: tuple[str, ...] = (
    "prediction_id",
    "tile_name",
    "tile_sha256",
    "label",
    "confidence",
    "needs_review",
    "model_version",
    "latency_ms",
    "created_at",
)
