from unittest.mock import Mock

import pytest

from app.database.documents import Document, to_metadata
from app.database.repository import DocumentRepository

def test_create_passes_explicit_vectors():
    collection = Mock()
    document = Document(content="Step one", embedding=[1.0, 0.0])
    assert DocumentRepository(collection).create(document) == document
    collection.add.assert_called_once_with(
        ids=[document.id], documents=[document.content],
        embeddings=[document.embedding], metadatas=[to_metadata(document)],
    )


def test_get_and_list_deserialize():
    document = Document(content="Step one", embedding=[1.0, 0.0])
    collection = Mock()
    collection.get.return_value = {
        "ids": [document.id], "documents": [document.content],
        "embeddings": [document.embedding], "metadatas": [to_metadata(document)],
    }
    repository = DocumentRepository(collection)
    assert repository.get(document.id) == document
    assert repository.list(limit=2, offset=1) == [document]
    assert collection.get.call_args.kwargs["offset"] == 1
    collection.get.return_value = {"ids": []}
    assert repository.get("absent") is None
    assert repository.delete("absent") is False
    collection.delete.assert_not_called()


@pytest.mark.parametrize("limit,offset", [(0, 0), (2, -1), (True, 0)])
def test_invalid_pagination(limit, offset):
    with pytest.raises(ValueError):
        DocumentRepository(Mock()).list(limit, offset)
