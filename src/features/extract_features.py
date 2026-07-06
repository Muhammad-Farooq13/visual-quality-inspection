"""
Classical computer-vision feature extraction for defect classification.

Why classical CV (HOG + LBP + intensity statistics) instead of a CNN?
Deep learning needs either a large labeled dataset or a pretrained backbone
to fine-tune -- neither is a good fit here: a from-scratch CNN on ~1,000
synthetic images would overfit or underfit unpredictably, and downloading a
pretrained backbone (e.g. from HuggingFace/torchvision model hubs) isn't
possible from this network-restricted environment. More importantly,
classical feature-based pipelines (HOG/LBP + a shallow classifier) are a
genuine, still-common choice in real manufacturing QA lines, especially on
edge/embedded hardware without a GPU, precisely because they're fast,
data-efficient, and fully interpretable -- which matters when a QA engineer
needs to understand *why* a part was flagged.

Features per image:
  - HOG (Histogram of Oriented Gradients): captures edge/gradient structure
    -- scratches and cracks show up as strong local gradient discontinuities.
  - LBP (Local Binary Patterns): captures local texture -- dents and
    discoloration disrupt the uniform brushed-metal texture pattern.
  - Global intensity statistics (mean, std, min, max, percentile spread):
    catches the darkness/contrast signature every defect type introduces.
"""

from __future__ import annotations

import logging

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern

logger = logging.getLogger(__name__)


def extract_features(image: np.ndarray, cfg: dict) -> np.ndarray:
    """Extract a single feature vector from an RGB or grayscale image array."""
    if image.ndim == 3:
        gray = rgb2gray(image)
    else:
        gray = image.astype(np.float64) / 255.0

    hog_cfg = cfg["features"]["hog"]
    hog_features = hog(
        gray,
        orientations=hog_cfg["orientations"],
        pixels_per_cell=tuple(hog_cfg["pixels_per_cell"]),
        cells_per_block=tuple(hog_cfg["cells_per_block"]),
        feature_vector=True,
    )

    lbp_cfg = cfg["features"]["lbp"]
    gray_uint8 = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(
        gray_uint8, P=lbp_cfg["n_points"], R=lbp_cfg["radius"], method=lbp_cfg["method"]
    )
    n_bins = lbp_cfg["n_points"] + 2
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    pixel_vals = gray.ravel() * 255.0
    intensity_stats = np.array(
        [
            pixel_vals.mean(),
            pixel_vals.std(),
            pixel_vals.min(),
            pixel_vals.max(),
            np.percentile(pixel_vals, 5),
            np.percentile(pixel_vals, 95),
        ]
    )

    return np.concatenate([hog_features, lbp_hist, intensity_stats])


def extract_features_with_hog_map(image: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Same as extract_features, but also returns the HOG cell-gradient map
    (used for localization/explainability, not for training)."""
    if image.ndim == 3:
        gray = rgb2gray(image)
    else:
        gray = image.astype(np.float64) / 255.0

    hog_cfg = cfg["features"]["hog"]
    hog_features, hog_image = hog(
        gray,
        orientations=hog_cfg["orientations"],
        pixels_per_cell=tuple(hog_cfg["pixels_per_cell"]),
        cells_per_block=tuple(hog_cfg["cells_per_block"]),
        feature_vector=True,
        visualize=True,
    )
    full_features = extract_features(image, cfg)
    return full_features, hog_image
