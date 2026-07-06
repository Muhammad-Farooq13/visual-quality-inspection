# Deployment Guide

## Local (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make pipeline   # generate images -> extract features -> train + select champion
make api        # FastAPI at http://localhost:8002
```

## Docker

```bash
docker build -f docker/Dockerfile -t visual-quality-inspection-api:latest .
docker run -p 8002:8002 visual-quality-inspection-api:latest
```

This pipeline needs **no external network access to build** beyond
`pip install` -- synthetic image generation, feature extraction, and model
training are all local, fast (classical CV features + a shallow
classifier, not a CNN — the full pipeline runs in under 90 seconds). The
Dockerfile runs the entire pipeline *inside* the image build, so the
champion model is baked into the image at build time rather than mounted
separately.

> As with the other repos in this series, `docker build` itself could not
> be executed in the sandbox used to author this project (no Docker daemon
> available there). Every individual command the Dockerfile runs was
> independently executed and verified in isolation in that same session
> (see `docs/ARCHITECTURE.md` for the exact simulation). CI does execute a
> real `docker build` and container smoke-test on every push.

## docker-compose

```bash
docker compose up --build
```

API available at http://localhost:8002/docs

## Cloud deployment targets

Standard stateless container exposing port 8002 with a `/health` endpoint
— deploys as-is to Render, Railway, AWS App Runner/ECS Fargate, Azure
Container Apps, or Google Cloud Run. No GPU required anywhere in this
pipeline, which keeps hosting cost low relative to a deep-learning-based
alternative.

## Retraining with a larger or real dataset

Swap `src/data_generation/generate_images.py`'s output for real labeled
inspection photos (same `good/` and `defective/` folder structure under
`data/synthetic_images/`, or point `configs/config.yaml`'s `synthetic_dir`
elsewhere) and re-run `make pipeline`. No other code changes needed — the
feature extraction and model comparison are dataset-agnostic.
