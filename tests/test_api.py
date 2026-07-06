import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _png_bytes(image_array) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image_array).save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@pytest.fixture(scope="module")
def defective_image_bytes():
    import glob

    path = sorted(glob.glob("data/synthetic_images/defective/*.png"))[0]
    with open(path, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def good_image_bytes():
    import glob

    path = sorted(glob.glob("data/synthetic_images/good/*.png"))[0]
    with open(path, "rb") as f:
        return f.read()


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "visual-quality-inspection-api"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert "champion_model" in body
    assert "metrics" in body


def test_inspect_defective_image(client, defective_image_bytes):
    response = client.post(
        "/inspect",
        files={"file": ("defective.png", defective_image_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["defective_probability"] <= 1.0
    assert body["risk_tier"] in {"low", "medium", "high"}
    assert len(body["heatmap_png_base64"]) > 0


def test_inspect_good_image(client, good_image_bytes):
    response = client.post("/inspect", files={"file": ("good.png", good_image_bytes, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["is_defective_predicted"], bool)


def test_inspect_rejects_non_image_content_type(client):
    response = client.post("/inspect", files={"file": ("notes.txt", b"hello world", "text/plain")})
    assert response.status_code == 422


def test_inspect_rejects_corrupt_image_data(client):
    response = client.post("/inspect", files={"file": ("fake.png", b"not a real png", "image/png")})
    assert response.status_code == 422


def test_inspect_handles_resizing_for_wrong_dimensions(client):
    """Images not matching the training image size should be resized, not rejected."""
    import numpy as np

    odd_size_image = (np.random.default_rng(1).random((256, 256, 3)) * 255).astype("uint8")
    png_bytes = _png_bytes(odd_size_image)

    response = client.post("/inspect", files={"file": ("odd.png", png_bytes, "image/png")})
    assert response.status_code == 200
