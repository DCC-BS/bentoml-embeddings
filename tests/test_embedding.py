import numpy as np
from service import Embeddings, EXAMPLE_INPUT



def test_encode():
    service = Embeddings()
    embeddings = service.encode_documents(EXAMPLE_INPUT)

    assert isinstance(embeddings, np.ndarray), "The embedding should be a numpy array."
    assert embeddings.shape[0] == len(EXAMPLE_INPUT), "The number of embeddings should be equal to the number of input sentences"
    assert embeddings.shape[1] == 1024, "The embedding dimension should be 1024"