import bentoml
import subprocess
import numpy as np

from service import EXAMPLE_INPUT


def test_embeddings_service_integration():
    with subprocess.Popen(
        ["bentoml", "serve", "service:Embeddings", "-p", "50001"]
    ) as server_proc:
        try:
            client = bentoml.SyncHTTPClient(
                "http://localhost:50001", server_ready_timeout=30
            )
            embeddings = client.encode_documents(documents=EXAMPLE_INPUT)

            assert isinstance(
                embeddings, np.ndarray
            ), "The response should be a string."
            assert embeddings.shape[0] == len(
                EXAMPLE_INPUT
            ), "Every sentences should be encoded"
            assert embeddings.shape[1] == 1024, "The embedding dimension should be 1024"

            embeddings = client.encode_query(query=EXAMPLE_INPUT[0])

            assert isinstance(
                embeddings, np.ndarray
            ), "The response should be a string."
            assert (
                embeddings.shape[0] == 1
            ), "The query should be a nd array with only one element"
            assert embeddings.shape[1] == 1024, "The embedding dimension should be 1024"
        finally:
            server_proc.terminate()
