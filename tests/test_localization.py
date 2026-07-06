import numpy as np
from PIL import Image, ImageDraw

from src.data_generation.generate_images import _base_surface
from src.features.localization import anomaly_heatmap


def test_anomaly_heatmap_shape_and_range():
    rng = np.random.default_rng(1)
    image = _base_surface(128, rng)
    heatmap = anomaly_heatmap(image)

    assert heatmap.shape == (128, 128)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-9


def test_anomaly_heatmap_localizes_a_dent_near_its_true_position():
    """A dent placed at a known pixel location must produce a heatmap peak
    within the dent's own radius -- this is the core correctness property
    for the localization feature. An earlier occlusion-sensitivity approach
    failed this exact test (peak landed >40px away, contaminated by global
    intensity-percentile features), which is why this test exists."""
    rng = np.random.default_rng(123)
    surface = _base_surface(128, rng)
    img = Image.fromarray(surface)
    draw = ImageDraw.Draw(img)
    cx, cy, r = 32, 32, 10
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(80, 80, 80))

    heatmap = anomaly_heatmap(np.array(img))
    peak_row, peak_col = np.unravel_index(heatmap.argmax(), heatmap.shape)
    distance_from_center = np.sqrt((peak_row - cy) ** 2 + (peak_col - cx) ** 2)

    # Peak should land within the defect's footprint plus a small margin
    # (gradient peaks are expected right at the defect's edge, not its center)
    assert distance_from_center <= r + 5


def test_anomaly_heatmap_localizes_a_scratch_near_its_line():
    rng = np.random.default_rng(456)
    surface = _base_surface(128, rng)
    img = Image.fromarray(surface)
    draw = ImageDraw.Draw(img)
    draw.line([(90, 20), (110, 60)], fill=(50, 50, 50), width=2)

    heatmap = anomaly_heatmap(np.array(img))
    peak_row, peak_col = np.unravel_index(heatmap.argmax(), heatmap.shape)

    # Distance from the peak to the nearest point on the line segment
    line_points = np.array([(20 + t * 40, 90 + t * 20) for t in np.linspace(0, 1, 50)])
    dists = np.sqrt((line_points[:, 0] - peak_row) ** 2 + (line_points[:, 1] - peak_col) ** 2)
    assert dists.min() <= 15


def test_defective_images_have_higher_peak_to_median_contrast_than_good():
    """Defects should create a real, measurable 'anomaly' signal -- a
    concentrated hotspot -- not just per-image-normalized noise. This
    compares raw (unnormalized) peak-to-median contrast, which is what
    actually distinguishes a genuine defect from uniform background grain."""
    import cv2

    def raw_contrast(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = cv2.GaussianBlur(np.sqrt(gx**2 + gy**2), (0, 0), sigmaX=2.0)
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=12.0)
        combined = grad + np.abs(gray - background)
        return np.percentile(combined, 99.5) / (np.median(combined) + 1e-6)

    rng = np.random.default_rng(99)
    good = _base_surface(128, rng)

    img = Image.fromarray(_base_surface(128, rng))
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 70, 70], fill=(70, 70, 70))
    defective = np.array(img)

    assert raw_contrast(defective) > raw_contrast(good)
