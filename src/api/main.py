"""
FastAPI serving layer for the visual quality inspection system.

Accepts an uploaded image, classifies it as good/defective using the
champion HOG+LBP+intensity-stats classifier, and returns both the
prediction and a statistical anomaly-map overlay (see
src/features/localization.py) so a QA reviewer can see *where* the surface
looks anomalous, not just a yes/no verdict.

Run:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8002 --reload
"""

from __future__ import annotations

import base64
import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.api.schemas import HealthResponse, InspectionResult, ModelInfoResponse
from src.features.extract_features import extract_features
from src.features.localization import anomaly_heatmap, overlay_heatmap

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
_state: dict = {}

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_model() -> None:
    cfg = _load_config()
    model_path = Path(cfg["artifacts"]["champion_model_path"])
    scaler_path = Path(cfg["artifacts"]["scaler_path"])
    metadata_path = Path(cfg["artifacts"]["metadata_path"])

    if not model_path.exists() or not metadata_path.exists():
        raise RuntimeError(
            f"Champion model artifacts not found at {model_path}. "
            "Run `make pipeline` first (generate images -> build features -> train)."
        )

    logger.info("Loading champion model from %s", model_path)
    _state["model"] = joblib.load(model_path)
    _state["scaler"] = joblib.load(scaler_path)
    with open(metadata_path, "r", encoding="utf-8") as fh:
        _state["metadata"] = json.load(fh)
    _state["config"] = cfg
    _state["threshold"] = cfg["api"]["decision_threshold"]
    logger.info(
        "Model loaded: %s (primary_metric=%s)",
        _state["metadata"]["champion_model"],
        _state["metadata"]["primary_metric"],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    _state.clear()


app = FastAPI(
    title="Visual Quality Inspection API",
    description=(
        "Classical computer-vision (HOG + LBP + intensity statistics) "
        "defect classifier for manufacturing surface inspection, with a "
        "statistical anomaly-map overlay for defect localization."
    ),
    version="1.0.0",
    contact={
        "name": "Muhammad Farooq Shafi",
        "email": "mfarooqsgafee333@gmail.com",
        "url": "https://www.linkedin.com/in/muhammadfarooqshafi/",
    },
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _risk_tier(prob: float) -> str:
    if prob < 0.3:
        return "low"
    if prob < 0.7:
        return "medium"
    return "high"


def _image_to_base64_png(image: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "visual-quality-inspection-api",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model_name=_state["metadata"]["champion_model"],
        model_version="1.0.0",
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["meta"])
def model_info() -> ModelInfoResponse:
    metadata = _state["metadata"]
    champion = metadata["champion_model"]
    return ModelInfoResponse(
        champion_model=champion,
        primary_metric=metadata["primary_metric"],
        metrics=metadata["all_results"][champion],
        decision_threshold=_state["threshold"],
    )


@app.post("/inspect", response_model=InspectionResult, tags=["inference"])
async def inspect(file: UploadFile = File(...)) -> InspectionResult:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported content type '{file.content_type}'. "
                f"Allowed: {ALLOWED_CONTENT_TYPES}"
            ),
        )

    raw_bytes = await file.read()
    try:
        image = np.array(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {exc}") from exc

    cfg = _state["config"]
    expected_size = cfg["data"]["image_size"]
    if image.shape[0] != expected_size or image.shape[1] != expected_size:
        image = np.array(Image.fromarray(image).resize((expected_size, expected_size)))

    features = extract_features(image, cfg)
    features_scaled = _state["scaler"].transform(features.reshape(1, -1))
    proba = float(_state["model"].predict_proba(features_scaled)[0, 1])

    heatmap = anomaly_heatmap(image)
    overlay = overlay_heatmap(image, heatmap)
    overlay_b64 = _image_to_base64_png(overlay)

    return InspectionResult(
        is_defective_predicted=bool(proba >= _state["threshold"]),
        defective_probability=round(proba, 6),
        risk_tier=_risk_tier(proba),
        decision_threshold=_state["threshold"],
        heatmap_png_base64=overlay_b64,
    )
