# API Reference

Interactive docs at `http://localhost:8002/docs` once running.

## `GET /health`

```json
{"status": "ok", "model_name": "random_forest", "model_version": "1.0.0"}
```

## `GET /model/info`

```json
{
  "champion_model": "random_forest",
  "primary_metric": "f1",
  "metrics": {
    "accuracy": 0.9708,
    "f1": 0.9700,
    "roc_auc": 0.9979,
    "precision_defective": 1.0,
    "recall_defective": 0.9417,
    "cv_best_score": 0.9710,
    "best_params": {"clf__max_depth": "20", "clf__min_samples_leaf": "2", "clf__n_estimators": "200"}
  },
  "decision_threshold": 0.5
}
```

## `POST /inspect`

Multipart file upload (`image/png`, `image/jpeg`, or `image/jpg`).

```bash
curl -X POST http://localhost:8002/inspect \
  -F "file=@part_photo.png;type=image/png"
```

**Response 200**
```json
{
  "is_defective_predicted": true,
  "defective_probability": 0.718018,
  "risk_tier": "high",
  "decision_threshold": 0.5,
  "heatmap_png_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

`heatmap_png_base64` is a base64-encoded PNG: the original image with a
statistical anomaly-map overlay (jet colormap) showing where the surface
looks anomalous. Decode and save it directly:

```python
import base64
with open("heatmap.png", "wb") as f:
    f.write(base64.b64decode(response.json()["heatmap_png_base64"]))
```

Images not matching the training resolution (128x128) are automatically
resized rather than rejected.

**422** — unsupported content type, or the file isn't a decodable image.

## Field reference

| Field | Type | Notes |
|---|---|---|
| `is_defective_predicted` | bool | `defective_probability >= decision_threshold` |
| `defective_probability` | float | 0-1, P(defective) from the champion classifier |
| `risk_tier` | enum | low (&lt;0.3), medium (0.3-0.7), high (&gt;0.7) |
| `decision_threshold` | float | Configurable in `configs/config.yaml` |
| `heatmap_png_base64` | string | Base64 PNG, anomaly-map overlay |
