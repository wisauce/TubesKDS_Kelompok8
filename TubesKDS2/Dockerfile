# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# System deps (Pillow + matplotlib runtime deps)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        curl \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY . /app

# Default port (can be overridden by platform)
ENV PORT=5050 \
    HOST=0.0.0.0

EXPOSE 5050

# Production WSGI server
# - `--preload` initializes the app once before forking workers (helps avoid duplicated model downloads)
# - `WEB_CONCURRENCY` can be set by platform; default 1 keeps memory low
CMD ["sh", "-c", "gunicorn --preload -w ${WEB_CONCURRENCY:-1} -b ${HOST}:${PORT} --timeout 120 api.app:app"]
