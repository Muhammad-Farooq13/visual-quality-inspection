"""Pydantic response schemas for the visual quality inspection API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class InspectionResult(BaseModel):
    is_defective_predicted: bool
    defective_probability: float = Field(..., ge=0, le=1)
    risk_tier: Literal["low", "medium", "high"]
    decision_threshold: float
    heatmap_png_base64: str = Field(
        ...,
        description="Base64-encoded PNG of the original image with an anomaly heatmap overlay",
    )


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_name: str
    model_version: str


class ModelInfoResponse(BaseModel):
    champion_model: str
    primary_metric: str
    metrics: dict
    decision_threshold: float
