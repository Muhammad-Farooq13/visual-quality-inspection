import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def project_config() -> dict:
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
