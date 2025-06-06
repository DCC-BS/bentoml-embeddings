FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y git build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

COPY . /app
WORKDIR /app

RUN uv sync && \
    MAX_JOBS=4 FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE uv pip install flash-attn==2.7.2.post1 --no-build-isolation

ENTRYPOINT ["uv", "run", "bentoml", "serve", "service:Embeddings", "-p", "50001"]