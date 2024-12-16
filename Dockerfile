FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.5.7 /uv /uvx /bin/

ADD . /app
WORKDIR /app

RUN uv sync
RUN uv pip install flash-attn==2.7.2.post1

ENTRYPOINT uv run bentoml serve service:Embeddings -p 50001