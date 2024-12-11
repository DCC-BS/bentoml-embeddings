from __future__ import annotations

import typing as t

import numpy as np
import bentoml

EMBEDDING_MODEL_ID = "jinaai/jina-embeddings-v3"
RANKER_TYPE = "cross-encoder"
LANGUAGE = "multi"
EXAMPLE_INPUT = ["Ich esse gerne Pizza."]

@bentoml.service(
    traffic={"timeout": 60},
    resources={"gpu": 1},
)
class Embeddings:

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
        self.encoder_model.max_seq_length = 8192

        self.ranker = Reranker(RANKER_TYPE, lang=LANGUAGE)



    @bentoml.api()
    def encode_documents(
        self,
        documents: t.List[str],
    ) -> np.ndarray:
        task = "retrieval.passage"
        return self.encoder_model.encode(documents, task=task, prompt_name=task)
    
    @bentoml.api()
    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        task = "retrieval.query"
        return self.encoder_model.encode([query], task=task, prompt_name=task)
    

    @bentoml.api()
    def rerank(
        self,
        sentences: t.List[str],
        query: str
    ) -> t.Dict[int, t.Tuple[str,float,int]]:
        ranked_results = self.ranker.rank(query=query, docs=sentences, doc_ids=list(range(0,len(sentences))))
        print(ranked_results)
        print(type(ranked_results))
        if len(sentences) == 1:
            result_json = {ranked_results.document.doc_id: (ranked_results.document.text, ranked_results.score, 0)}
        else:
            result_json = {result.doc_id: (result.text, result.score, result.rank) for result in ranked_results}

        print(result_json)
        return result_json
