FROM python:3.11-slim AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY src ./src

RUN useradd --create-home --uid 10001 smio \
    && mkdir -p /app/models \
    && chown smio:smio /app /app/models
USER smio

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]

FROM runtime-base AS cloud-run

FROM runtime-base AS local
COPY --chown=smio:smio models/ner-smio ./models/ner-smio
COPY --chown=smio:smio models/distilbert_deployed ./models/distilbert_deployed