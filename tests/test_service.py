import numpy as np
from service import Embeddings, EXAMPLE_INPUT


def test_encode_documents():
    service = Embeddings()
    embeddings = service.encode_documents(EXAMPLE_INPUT)

    assert isinstance(embeddings, np.ndarray), "The embedding should be a numpy array."
    assert embeddings.shape[0] == len(
        EXAMPLE_INPUT
    ), "The number of embeddings should be equal to the number of input sentences"
    assert embeddings.shape[1] == 1024, "The embedding dimension should be 1024"


def test_encode_query():
    service = Embeddings()
    embeddings = service.encode_query(EXAMPLE_INPUT[0])

    assert isinstance(embeddings, np.ndarray), "The embedding should be a numpy array."
    assert (
        embeddings.shape[0] == 1
    ), "The query should be a nd array with only one element"
    assert embeddings.shape[1] == 1024, "The embedding dimension should be 1024"


def test_rerank_valid_input():
    """Test rerank function with valid input"""
    service = Embeddings()

    # Test case with multiple sentences
    sentences = [
        "Hello world",
        "Python ist eine Programmiersprache. Ich möchte sie heute testen.",
        "Unit tests are important for testing.",
    ]
    query = "testing"

    result = service.rerank(documents=sentences, query=query)

    assert isinstance(result, dict), "Result should be a dictionary"
    assert len(result) == len(
        sentences
    ), "Result should contain same number of entries as input"

    # Check structure of each result entry
    for idx, (sentence, score, rank) in result.items():
        assert isinstance(idx, int), "Key should be integer"
        assert isinstance(sentence, str), "First tuple element should be string"
        assert isinstance(score, float), "Second tuple element should be float"
        assert isinstance(rank, int), "Third tuple element should be integer"
        assert 1 <= rank < len(sentences) + 1, "Rank should be within valid range"

    # Check if ranking is correct
    sorted_ranks = sorted(result.items(), key=lambda x: x[1][2])
    for i in range(len(sorted_ranks) - 1):
        assert (
            sorted_ranks[i][1][2] < sorted_ranks[i + 1][1][2]
        ), "Ranks should be in ascending order"
    assert (
        sorted_ranks[0][1][0] == sentences[-1]
    ), "Wrong sentence rank"  # Assuming last sentence is most relevant to the query
    assert (
        sorted_ranks[1][1][0] == sentences[1]
    ), (
        "Wrong sentence rank"
    )  # Assuming second sentence is second most relevant to the query
    assert (
        sorted_ranks[2][1][0] == sentences[0]
    ), "Wrong sentence rank"  # Assuming first sentence is most irrelevant sentence


def test_rerank_single_sentence():
    """Test rerank function with single sentence input"""
    service = Embeddings()

    sentences = ["Single test sentence"]
    query = "test"

    result = service.rerank(documents=sentences, query=query)

    assert len(result) == 1, "Result should contain exactly one entry"
    assert list(result.values())[0][2] == 0, "Rank should be 0 for single sentence"


def test_rerank_special_characters():
    """Test rerank function with special characters"""
    service = Embeddings()

    sentences = ["Hello! @#$%", "Test with \n newline", "Unicode: 你好"]
    query = "Special @#$ chars"

    result = service.rerank(documents=sentences, query=query)
    assert len(result) == len(sentences), "Should handle special characters correctly"
