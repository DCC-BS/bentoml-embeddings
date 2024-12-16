from starlette.testclient import TestClient
from service import Embeddings, EXAMPLE_INPUT
import numpy as np
import json
import pytest


app = Embeddings.to_asgi()

def test_http_lifespan():
    with TestClient(app=app) as test_client:
        response = test_client.post(
            "/encode_documents", json={"documents": EXAMPLE_INPUT}
        )
        embeddings = np.array(json.loads(response.content))
        assert response.status_code == 200
        assert embeddings.shape[0] == len(
            EXAMPLE_INPUT
        ), "Every sentences should be encoded"
        assert embeddings.shape[1] == 1024, "The embedding dimension should be 1024"

@pytest.mark.parametrize(
    "documents",
    [
        [], # Empty sentences list
        None, # None sentences
        [None], # None sentence
        [""], # Empty string
        ["", ""], # Empty strings
        [1, 2, 3], # Non-string sentences

    ],
)
def test_encode_documents_invalid_inputs(documents):
    with TestClient(app) as test_client:
        response = test_client.post("/encode_documents", json={"documents": documents})
        assert response.status_code == 400, "Invalid input should return a 400 error"

def test_encode_documents_invalid_inputs_long_stuff():
    with TestClient(app) as test_client:
        documents = ["sentence"]* 33, # too many documents
        response = test_client.post("/encode_documents", json={"documents": documents})
        assert response.status_code == 400, "Invalid input should return a 400 error"

        documents = ["sentence" * 10_000], # too long document string
        response = test_client.post("/encode_documents", json={"documents": documents})
        assert response.status_code == 400, "Invalid input should return a 400 error"

def test_encode_documents():
    with TestClient(app) as test_client:
        response = test_client.post(
            "/encode_documents", json={"documents": EXAMPLE_INPUT}
        )
        embeddings = np.array(json.loads(response.content))
        assert response.status_code == 200
        assert embeddings.shape[0] == len(
            EXAMPLE_INPUT
        ), "Every sentences should be encoded"
        assert embeddings.shape[1] == 1024, "The embedding dimension should be 1024"

        response = test_client.post("/encode_documents")
        assert response.status_code == 400, "Documents is required"

        response = test_client.get("/encode_documents")
        assert response.status_code == 405, "GET is not allowed"

def test_encode_query():
    with TestClient(app) as test_client:
        response = test_client.post(
            "/encode_query", json={"query": "Test query"}
        )
        embeddings = np.array(json.loads(response.content))
        assert response.status_code == 200
        assert embeddings.shape[0] == len(
            EXAMPLE_INPUT
        ), "Every sentences should be encoded"
        assert embeddings.shape[1] == 1024, "The embedding dimension should be 1024"
    

@pytest.mark.parametrize(
    "query",
    [
        [], # Invalid type
        None, # None sentences
        "", # Empty query
        1, # Invalid type
    ],
)
def test_encode_query_invalid_input(query):
    with TestClient(app) as test_client:
        response = test_client.post(
            "/encode_query", json={"query": query}
        )
        assert response.status_code == 400, "Invalid input should return a 400 status code"

def test_encode_query_invalid_input_long_stuff():
    with TestClient(app) as test_client:
        query = "sentence" * 10_000
        response = test_client.post(
            "/encode_query", json={"query": query}
        )
        assert response.status_code == 400, "Invalid input should return a 400 status code"


def test_rank_documents():
    with TestClient(app) as test_client:
        response = test_client.post("/rerank", json={"documents": EXAMPLE_INPUT, "query": "query"})
        assert response.status_code == 200

        response = test_client.post("/rerank", json={"documents": EXAMPLE_INPUT})
        assert response.status_code == 400, "Query is required"

        response = test_client.post("/rerank", json={"query": "EXAMPLE_INPUT"})
        assert response.status_code == 400, "Documents is required"



@pytest.mark.parametrize(
    "invalid_inputs",
    [
        ([], "Valid query"), # Empty sentences list
        (None,"Valid query"), # None sentences
        ([""],"Valid query"), # Empty string
        ([1, 2, 3], "Valid query"), # Non-string sentences
    ],
)
def test_rank_documents_invalid_inputs(invalid_inputs):
    document, query = invalid_inputs
    with TestClient(app) as test_client:
        response = test_client.post("/rerank", json={"documents": document, "query": query})
        assert response.status_code == 400, "Invalid input should return a 400 error"

def test_rank_documents_invalid_inputs_long_stuff():
        with TestClient(app) as test_client:
            documents, query = (["sentence"]* 33, "Valid query")
            response = test_client.post("/rerank", json={"documents": documents, "query": query})
            assert response.status_code == 400, "Invalid input should return a 400 error"
            
            documents, query = (["sentence" * 10_000], "Valid query")
            response = test_client.post("/rerank", json={"documents": documents, "query": query})
            assert response.status_code == 400, "Invalid input should return a 400 error"