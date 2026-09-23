FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models/distilbert_deployed ./models/distilbert_deployed
COPY models/ner-smio ./models/ner-smio

RUN useradd --create-home --uid 10001 smio \
    && chown -R smio:smio /app
USER smio

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]