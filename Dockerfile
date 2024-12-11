FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.5.7 /uv /uvx /bin/

ADD . /app
WORKDIR /app

RUN uv sync
RUN uv sync --build
RUN uv sync --compile
ENTRYPOINT uv run bentoml serve service:Embeddings -p 50001