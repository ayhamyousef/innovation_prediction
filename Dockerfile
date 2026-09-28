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
RUN mkdir -p data/raw data/processed data/cache results

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# No default stage: the pipeline runs in the order given in the README, and each
# stage is selected explicitly, e.g.
#   docker run --rm -e ODP_API_KEY=... -v "$PWD/data:/app/data" IMAGE \
#     scripts/01_fetch_data.py --start-year 2002 --end-year 2022
# docker-compose.yml defines one service per stage.
ENTRYPOINT ["python"]
CMD ["--version"]
