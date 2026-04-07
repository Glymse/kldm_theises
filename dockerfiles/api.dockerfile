ARG UV_VERSION=latest

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.11-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv

COPY --from=uv /uv /uvx /usr/local/bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY README.md ./

RUN uv sync --frozen
RUN uv pip install "setuptools>=70,<82" "omegaconf>=2.3,<2.4" "hydra-core>=1.3,<1.4" "emmet-core>=0.84,<0.85"

RUN arch="$(uname -m)" && \
    if [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; then \
      uv pip install --no-build-isolation --no-binary torch-scatter "torch-scatter==2.1.2"; \
    fi

ENTRYPOINT ["uv", "run", "uvicorn", "src.kldm.api:app", "--host", "0.0.0.0", "--port", "8000"]
