# Servinge Embeddings and Rankers with BentoML

[BentoML](https://www.bentoml.com/) API for [Jina v3 multilingual embeddings](https://huggingface.co/jinaai/jina-embeddings-v3) and [cross encoder reranker](https://huggingface.co/corrius/cross-encoder-mmarco-mMiniLMv2-L12-H384-v1) using the [reranker wrapper](https://github.com/AnswerDotAI/rerankers).

## Preprequisites

[Install astral uv](https://docs.astral.sh/uv/getting-started/installation/).

## Run the BentoML Service locally

`uv run bento serve service:Embeddings`

This will run the BentoML API with a Swagger Documentation up and running.

## Run as docker container

docker compose up --build

## Push docker image to quay.io

docker push quay.io/ktbs/fd-itbs-dms/embeddings

## Change the embedding or reranker model

In the `service.py` file:

For the embedding model, change `EMBEDDING_MODEL_ID` to your hugging face sentence transformer compatible embedding model.

For the reranker change `RANKER_TYPE` and `LANGUAGE` according to the [docs of the rerankers library](https://github.com/AnswerDotAI/rerankers).

## Run tests

uv run pytest
