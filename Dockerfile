FROM python:3.12-slim AS base

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home appuser

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache "." \
    && mkdir -p /app/logs /app/data/cache /app/data/parquet \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501
