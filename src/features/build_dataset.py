"""
Builds the model-ready feature matrix from the synthetic image dataset.

Run:
    python -m src.features.build_dataset --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from src.features.extract_features import extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _parse_defect_type(filename: str) -> str:
    match = re.match(r"defective_\d+_(\w+)\.png", filename)
    return match.group(1) if match else "unknown"


def build_dataset(cfg: dict) -> pd.DataFrame:
    synthetic_dir = Path(cfg["data"]["synthetic_dir"])
    good_paths = sorted((synthetic_dir / "good").glob("*.png"))
    defective_paths = sorted((synthetic_dir / "defective").glob("*.png"))

    rows = []
    logger.info("Extracting features from %d good images", len(good_paths))
    for path in good_paths:
        img = np.array(Image.open(path).convert("RGB"))
        features = extract_features(img, cfg)
        rows.append(
            {
                "image_path": str(path),
                "is_defective": 0,
                "defect_type": "none",
                "features": features,
            }
        )

    logger.info("Extracting features from %d defective images", len(defective_paths))
    for path in defective_paths:
        img = np.array(Image.open(path).convert("RGB"))
        features = extract_features(img, cfg)
        rows.append(
            {
                "image_path": str(path),
                "is_defective": 1,
                "defect_type": _parse_defect_type(path.name),
                "features": features,
            }
        )

    df = pd.DataFrame(rows)
    logger.info(
        "Built feature dataset: %d rows, %d feature dims",
        len(df),
        len(df["features"].iloc[0]),
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature dataset from synthetic images.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output", type=str, default="data/processed_features.npz")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    df = build_dataset(cfg)

    X = np.stack(df["features"].values)
    y = df["is_defective"].values
    defect_types = df["defect_type"].values
    image_paths = df["image_path"].values

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X=X, y=y, defect_types=defect_types, image_paths=image_paths)
    logger.info("Saved feature dataset to %s (X shape: %s)", args.output, X.shape)


if __name__ == "__main__":
    main()
