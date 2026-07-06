# =============================================================================
# Visual Quality Inspection API - Production Docker Image
#
# Like the marketing-attribution-warehouse repo, this pipeline needs no
# external network access to build beyond pip install -- synthetic image
# generation, feature extraction, and model training are all local and
# fast (classical CV features + a shallow classifier, not a CNN). The image
# runs the full pipeline at build time, so if the trained model doesn't
# clear a minimum quality bar, the build fails rather than shipping a bad
# model silently.
# =============================================================================

FROM python:3.12-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY configs/ ./configs/
COPY src/ ./src/

# Run the full pipeline at build time: generate synthetic images, extract
# features, train + select the champion model.
ENV PATH=/root/.local/bin:$PATH PYTHONPATH=/build
RUN python -m src.data_generation.generate_images --config configs/config.yaml \
    && python -m src.features.build_dataset --config configs/config.yaml \
    && python -m src.models.train --config configs/config.yaml

# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder /root/.local /home/appuser/.local
COPY --from=builder /build/artifacts/models/ ./artifacts/models/
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY src/ ./src/
COPY configs/ ./configs/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8002"]
