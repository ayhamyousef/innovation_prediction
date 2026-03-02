FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/

# Data / output volumes
RUN mkdir -p data/raw data/processed data/cache models/runs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python"]
CMD ["scripts/05_train_transformer.py"]
