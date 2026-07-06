"""
Synthetic industrial surface image generator.

Why synthetic? Real manufacturing defect datasets (e.g. MVTec AD) are
research-license-only and not redistributable, and downloading them requires
hosts not reachable from this environment. This generator produces
procedurally-textured "product surface" images -- metal-like brushed
texture with realistic per-pixel noise -- and injects one of four defect
types (scratch, dent, discoloration, crack) into the "defective" class,
each with randomized position, size, and orientation so the dataset isn't
trivially memorizable by pixel position alone.

Run:
    python -m src.data_generation.generate_images --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _base_surface(size: int, rng: np.random.Generator) -> np.ndarray:
    """Procedural brushed-metal-like surface: directional noise + base tone."""
    base_tone = rng.uniform(150, 190)
    surface = np.full((size, size), base_tone, dtype=np.float64)

    # Directional (horizontal) brushed-metal streaks
    row_noise = rng.normal(0, 6, size=(size, 1))
    surface += row_noise

    # Fine per-pixel grain
    grain = rng.normal(0, 4, size=(size, size))
    surface += grain

    surface = np.clip(surface, 0, 255)
    img = Image.fromarray(surface.astype(np.uint8), mode="L").convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    return np.array(img)


def _add_scratch(img: Image.Image, rng: np.random.Generator) -> None:
    draw = ImageDraw.Draw(img)
    size = img.size[0]
    x1, y1 = rng.integers(0, size, size=2)
    length = rng.integers(size // 4, size // 2)
    angle = rng.uniform(0, 2 * np.pi)
    x2 = int(np.clip(x1 + length * np.cos(angle), 0, size - 1))
    y2 = int(np.clip(y1 + length * np.sin(angle), 0, size - 1))
    darkness = int(rng.integers(40, 90))
    width = int(rng.integers(1, 3))
    draw.line([(x1, y1), (x2, y2)], fill=(darkness, darkness, darkness), width=width)


def _add_dent(img: Image.Image, rng: np.random.Generator) -> None:
    draw = ImageDraw.Draw(img)
    size = img.size[0]
    cx, cy = rng.integers(size // 4, 3 * size // 4, size=2)
    r = int(rng.integers(size // 16, size // 8))
    darkness = int(rng.integers(60, 110))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(darkness, darkness, darkness))
    img_blurred = img.filter(ImageFilter.GaussianBlur(radius=2.0))
    img.paste(img_blurred, (0, 0))


def _add_discoloration(img: Image.Image, rng: np.random.Generator) -> None:
    size = img.size[0]
    arr = np.array(img).astype(np.float64)
    cx, cy = rng.integers(size // 4, 3 * size // 4, size=2)
    r = rng.integers(size // 8, size // 5)
    yy, xx = np.ogrid[:size, :size]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    tint = rng.choice([(1.0, 0.7, 0.5), (0.6, 0.6, 1.0), (0.7, 1.0, 0.7)])
    for c in range(3):
        arr[..., c][mask] *= tint[c]
    arr = np.clip(arr, 0, 255)
    img.paste(Image.fromarray(arr.astype(np.uint8)), (0, 0))


def _add_crack(img: Image.Image, rng: np.random.Generator) -> None:
    draw = ImageDraw.Draw(img)
    size = img.size[0]
    x, y = rng.integers(size // 4, 3 * size // 4, size=2)
    n_segments = rng.integers(4, 8)
    darkness = int(rng.integers(30, 70))
    points = [(x, y)]
    angle = rng.uniform(0, 2 * np.pi)
    for _ in range(n_segments):
        seg_len = rng.integers(6, 16)
        angle += rng.uniform(-1.0, 1.0)
        nx = int(np.clip(points[-1][0] + seg_len * np.cos(angle), 0, size - 1))
        ny = int(np.clip(points[-1][1] + seg_len * np.sin(angle), 0, size - 1))
        points.append((nx, ny))
    draw.line(points, fill=(darkness, darkness, darkness), width=1)


DEFECT_FUNCS = {
    "scratch": _add_scratch,
    "dent": _add_dent,
    "discoloration": _add_discoloration,
    "crack": _add_crack,
}


def generate_dataset(cfg: dict) -> None:
    seed = cfg["project"]["seed"]
    rng = np.random.default_rng(seed)
    size = cfg["data"]["image_size"]
    n_per_class = cfg["data"]["n_images_per_class"]
    out_dir = Path(cfg["data"]["synthetic_dir"])

    good_dir = out_dir / "good"
    defective_dir = out_dir / "defective"
    good_dir.mkdir(parents=True, exist_ok=True)
    defective_dir.mkdir(parents=True, exist_ok=True)

    defect_names = [d["name"] for d in cfg["defect_types"]]
    defect_weights = np.array([d["weight"] for d in cfg["defect_types"]])
    defect_weights = defect_weights / defect_weights.sum()

    logger.info("Generating %d good images", n_per_class)
    for i in range(n_per_class):
        surface = _base_surface(size, rng)
        img = Image.fromarray(surface)
        img.save(good_dir / f"good_{i:04d}.png")

    logger.info("Generating %d defective images across %s", n_per_class, defect_names)
    labels = []
    for i in range(n_per_class):
        surface = _base_surface(size, rng)
        img = Image.fromarray(surface)
        defect_type = rng.choice(defect_names, p=defect_weights)
        DEFECT_FUNCS[defect_type](img, rng)
        img.save(defective_dir / f"defective_{i:04d}_{defect_type}.png")
        labels.append(defect_type)

    from collections import Counter

    logger.info("Defect type distribution: %s", dict(Counter(labels)))
    logger.info("Saved synthetic images to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic surface inspection images.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    generate_dataset(cfg)


if __name__ == "__main__":
    main()
