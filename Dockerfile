# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Churn Intelligence Platform — single image that can serve either the FastAPI
# API (default) or the Streamlit dashboard (override the command).
# ---------------------------------------------------------------------------
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp1 is required by xgboost / shap at runtime; curl powers the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY . .

# Bake model artifacts + a seed scoring history into the image so it runs out of
# the box. tune=False keeps the build fast; retrain with the full search at
# runtime if desired.
RUN python -m src.generate_data \
    && python -c "from src.model import run_pipeline; run_pipeline(tune=False)" \
    && python -m src.explain \
    && python -m src.simulate --batches 5 --batch-size 30 --reset

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]
# Default command: serve the API. docker-compose overrides this for the dashboard.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
