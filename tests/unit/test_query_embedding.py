import pytest

@pytest.mark.unit
def test_query_embedding(query):
    top_k = 2
    embeddings = [[0.1], [0.2], [0.3]]

    result = embed_query(query, embeddings, top_k)

    assert result == [[0.1], [0.2]]
