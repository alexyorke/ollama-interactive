FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates git ripgrep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY ollama_code ./ollama_code
COPY tests ./tests
COPY docker/entrypoint.sh /usr/local/bin/ollama-code-docker

RUN chmod +x /usr/local/bin/ollama-code-docker \
    && python -m pip install --upgrade pip \
    && python -m pip install -e .

RUN mkdir -p /workspace

ENTRYPOINT ["ollama-code-docker"]
