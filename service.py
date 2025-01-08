from __future__ import annotations
import os
import typing as t

import numpy as np
import bentoml
from pydantic import Field
from annotated_types import MinLen, MaxLen
from typing import Annotated
from torch import Tensor


EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "jinaai/jina-embeddings-v3")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", 8192))
MAX_DOCS = int(os.getenv("MAX_DOCS", 32))
RANKER_TYPE = os.getenv("TIMEOUT", "cross-encoder")
LANGUAGE = os.getenv("LANGUAGE", "multi")
# The combination of RANKER_TYPE and LANGUAGE determines the specific ranker model.
# See https://github.com/AnswerDotAI/rerankers/blob/main/rerankers/reranker.py

EXAMPLE_INPUT = ["Ich esse gerne Pizza."]

ValidString = Annotated[str, MinLen(1)]
ValidStringList = Annotated[t.List[ValidString], MinLen(1), MaxLen(MAX_DOCS)]


@bentoml.service(
    traffic={"timeout": 60},
)
class Embeddings:
    """
    BentoML service class for handling embeddings using the JinaAI jina-embeddings-v3 model.
    
    This class defines a service that uses GPU resources and has a timeout of 60 seconds.
    The service is designed to process text inputs and generate corresponding embeddings.
    
    Attributes:
        EMBEDDING_MODEL_ID (str): The identifier for the embedding model used by this service.
        RANKER_TYPE (str): Specifies the type of ranker used, in this case 'cross-encoder'.
        LANGUAGE (str): Indicates the language support of the model, here set to 'multi' for multilingual support. The combination of LANGUAGE and RANKER_TYPE leads to the ranker model in this case corrius/cross-encoder-mmarco-mMiniLMv2-L12-H384-v1
        EXAMPLE_INPUT (List[str]): An example input text that can be used with this service.
    """

    def __init__(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer
        from rerankers import Reranker

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder_model = SentenceTransformer(
            EMBEDDING_MODEL_ID,
            trust_remote_code=True,
            device=self.device
        )

        # control your input sequence length up to 8192
        self.encoder_model.max_seq_length = MAX_SEQ_LENGTH

        self.ranker = Reranker(RANKER_TYPE, lang=LANGUAGE)



    @bentoml.api()
    def encode_documents(
        self,
        documents: ValidStringList,
    ) -> np.ndarray:
        """
        Encodes a list of documents into embeddings.

        Args:
            documents (List[str]): A list of document strings to be encoded.

        Returns:
            np.ndarray: An array containing the embeddings for each input document.
        """
        task = "retrieval.passage"
        return self.encoder_model.encode(documents, task=task, prompt_name=task)
    
    @bentoml.api()
    def encode_queries(
        self,
        queries: ValidStringList
    ) -> np.ndarray:
        """
        Encodes a list of query strings into query embeddings.

        Args:
            query (List[str]): A list of query strings to be encoded.

        Returns:
            np.ndarray: An array containing the query embeddings for the input queries.
        """
        task = "retrieval.query"
        return self.encoder_model.encode(queries, task=task, prompt_name=task)
    

    @bentoml.api()
    def rerank(
        self,
        documents: ValidStringList = Field(description="List of documents to be reranked"),
        query: ValidString = Field(description="Query string used for ranking documents"),
    ) -> t.Dict[int, t.Tuple[str,float,int]]:
        """
        Reranks a list of documents based on their relevance to a given query.

        Args:
            documents (List[str]): A list of documents to be reranked.
            query (str): The query string used for ranking the documents.

        Returns:
            Dict[int, Tuple[str, float, int]]: A dictionary mapping original indices
                to tuples containing the document text, relevance score, and rank index.
        """
        ranked_results = self.ranker.rank(query=query, docs=documents, doc_ids=list(range(0,len(documents))))
        if len(documents) == 1:
            result_json = {ranked_results.document.doc_id: (ranked_results.document.text, ranked_results.score, 0)}
        else:
            result_json = {result.doc_id: (result.text, result.score, result.rank) for result in ranked_results}
        return result_json

    @bentoml.api
    def tokenize(
        self,
        documents: ValidStringList = Field(description="Documents that should be tokenized")
    ) -> t.List[t.List[int]]:
        tokens: Tensor = self.encoder_model.tokenize(texts=documents)["input_ids"]
        return tokens.tolist()
