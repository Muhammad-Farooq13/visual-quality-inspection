"""
Defect localization via statistical anomaly mapping.

An earlier version of this module tried occlusion sensitivity (tile the
image, re-run the classifier with each tile blurred out, measure the
prediction-probability drop). That approach was tested against a controlled
image with a defect at a known pixel location and it failed: the model's
feature vector includes global order-statistics (min/max/percentiles of
pixel intensity), which are highly sensitive to *any* single extreme pixel
anywhere in the image -- not specifically the defect region. Occluding an
unrelated tile that happened to contain the image's darkest grain pixel
shifted those global statistics more than occluding the actual defect did,
producing a localization heatmap that pointed at the wrong place. That
failure is captured in git history / dev notes rather than hidden.

This module instead computes localization directly from image statistics,
decoupled entirely from the classifier:
  - Gradient magnitude (Sobel): highlights thin, sharp features -- scratches
    and cracks are exactly this.
  - Local intensity deviation from a heavily-blurred version of the same
    image (i.e. deviation from the local "background"): highlights dents
    (locally darker blobs) and discoloration (locally shifted color/tone)
    that don't necessarily produce strong gradients.

The two channels are combined and normalized into a single [0,1] heatmap.
This is intentionally presented as a statistical anomaly map, not a
"classifier explanation" -- it answers "where does this surface look
anomalous," which is what a QA reviewer actually needs, without overclaiming
insight into the RandomForest's internal decision process.
"""

from __future__ import annotations

import cv2
import numpy as np


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def anomaly_heatmap(
    image: np.ndarray,
    background_blur_sigma: float = 12.0,
    gradient_smooth_sigma: float = 2.0,
    gradient_weight: float = 0.5,
    intensity_weight: float = 0.5,
) -> np.ndarray:
    """
    Args:
        image: HxWx3 uint8 RGB array.
        background_blur_sigma: how heavily to blur the image to estimate the
            local "background" tone for intensity-deviation comparison.
            Larger = only large-scale tone shifts count as anomalous.
        gradient_smooth_sigma: smooths raw per-pixel gradient magnitude into
            small blobs rather than a thin one-pixel-wide line, so it reads
            as a heatmap region rather than a wireframe outline.
        gradient_weight, intensity_weight: relative contribution of each
            signal to the combined heatmap (should sum to ~1.0).

    Returns:
        HxW float array in [0, 1].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag_smoothed = cv2.GaussianBlur(grad_mag, (0, 0), sigmaX=gradient_smooth_sigma)

    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=background_blur_sigma)
    intensity_dev = np.abs(gray - background)

    combined = gradient_weight * _normalize(grad_mag_smoothed) + intensity_weight * _normalize(
        intensity_dev
    )
    return _normalize(combined)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a normalized [0,1] heatmap onto an RGB image with a jet colormap."""
    import matplotlib.cm as cm

    colored = (cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
    base = image.astype(np.float64)
    overlay = (1 - alpha) * base + alpha * colored.astype(np.float64)
    return np.clip(overlay, 0, 255).astype(np.uint8)
