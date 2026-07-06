.PHONY: help install data features train pipeline test lint format api docker-build docker-up docker-down clean

PYTHON := python3
CONFIG := configs/config.yaml

help:
	@echo "Visual Quality Inspection - available targets:"
	@echo "  install      Install dependencies"
	@echo "  data         Generate synthetic surface inspection images"
	@echo "  features     Extract HOG+LBP+intensity features from images"
	@echo "  train        Train and compare classifiers, save champion"
	@echo "  pipeline     data -> features -> train, end to end"
	@echo "  test         Run the pytest suite (requires pipeline to have run)"
	@echo "  lint         Run ruff lint checks"
	@echo "  format       Auto-format code with black"
	@echo "  api          Run the FastAPI inspection service locally"
	@echo "  docker-build Build the Docker image (runs the full pipeline at build time)"
	@echo "  docker-up    Start the API via docker-compose"
	@echo "  docker-down  Stop docker-compose services"
	@echo "  clean        Remove generated artifacts and caches"

install:
	pip install --break-system-packages -r requirements.txt

data:
	$(PYTHON) -m src.data_generation.generate_images --config $(CONFIG)

features:
	$(PYTHON) -m src.features.build_dataset --config $(CONFIG)

train:
	$(PYTHON) -m src.models.train --config $(CONFIG)

pipeline: data features train

test:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	black src/ tests/

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8002 --reload

docker-build:
	docker build -f docker/Dockerfile -t visual-quality-inspection-api:latest .

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
