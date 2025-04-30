FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.7.1 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y git build-essential

ADD . /app
WORKDIR /app

RUN uv sync
RUN MAX_JOBS=4 FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE uv pip install flash-attn==2.7.2.post1 --no-build-isolation

ENTRYPOINT uv run bentoml serve service:Embeddings -p 50001