from starlette.testclient import TestClient
from service import Embeddings, EXAMPLE_INPUT
import pytest

def test_request():
    app = Embeddings.to_asgi()

    with TestClient(app=app) as test_client:
        response = test_client.post("/encode_documents", json={"documents": EXAMPLE_INPUT})
        embeddings = response.text
        assert response.status_code == 200
        assert embeddings.shape()[0] == len(EXAMPLE_INPUT), "Every sentences should be encoded"