from app.database.documents import Document, from_record, to_metadata

def test_document_defaults_are_independent():
    first = Document(content="First", embedding=[1.0, 0.0])
    second = Document(content="Second", embedding=[0.0, 1.0])
    assert first.id != second.id
    assert first.created_at

def test_metadata_round_trip():
    document = Document(
        content="Check calibration", embedding=[1.0, 0.0],
        source_id="protocol-1", storage_key="documents/source.txt",
        filename="source.txt", source="source.txt", chunk_index=2,
        start=10, end=27, embedding_model="model.gguf",
    )
    metadata = to_metadata(document)
    assert "embedding" not in metadata
    assert "content" not in metadata
    restored = from_record(document.id, document.content, document.embedding, metadata)
    assert restored == document
    assert "embedding" not in document.summary()
