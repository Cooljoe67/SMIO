FROM python:3.11-slim AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models/ner-smio/config.json ./models/ner-smio/config.json
COPY models/ner-smio/model.safetensors ./models/ner-smio/model.safetensors
COPY models/ner-smio/special_tokens_map.json ./models/ner-smio/special_tokens_map.json
COPY models/ner-smio/tokenizer.json ./models/ner-smio/tokenizer.json
COPY models/ner-smio/tokenizer_config.json ./models/ner-smio/tokenizer_config.json
COPY models/ner-smio/vocab.txt ./models/ner-smio/vocab.txt

RUN useradd --create-home --uid 10001 smio \
    && chown -R smio:smio /app
USER smio

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]

FROM runtime-base AS cloud-run

FROM runtime-base AS local
COPY --chown=smio:smio models/distilbert_deployed ./models/distilbert_deployed