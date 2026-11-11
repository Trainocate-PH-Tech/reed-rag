from app.processors.base import BaseProcessor
from app.processors.text import TextProcessor
from app.processors.text_file import TextFileProcessor
from app.providers.local import LocalEmbeddingProvider


PROCESSORS: dict[str, type[BaseProcessor]] = {
    "text": TextProcessor
}

def embed_text_file(
    path: str,
    chunk_size: int,
    stride: int,
    model: str,
    batch_size: int,
    provider_factory,
    context_size=2048,
    gpu_layers=0
) -> dict:
    document = process_text_file(path, chunk_size, stride)
    chunks = [chunk for chunk in document["chunks"] if chunk["text"].strip()]

    factory = provider_factory if provider_factory is not None else LocalEmbeddingProvider

    provider = factory(
        model_path=model,
        context_size=context_size,
        gpu_layers=gpu_layers
    )

    records = []
    dimensions = None

    DOCUMENT_PREFIX = "title: none | text: "
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vectors = provider.embed([DOCUMENT_PREFIX + chunk["text"] for chunk in batch])
        if len(vectors) != len(batch):
            raise ValueError("Embedding count mismatch")
        for chunk, vector in zip(batch, vectors):
            if not vector:
                raise ValueError("Empty embedding vector")
            if dimensions is None:
                dimensions = len(vector)
            elif len(vector) != dimensions:
                raise ValueError("Inconsistent vector dimensions")
            records.append({**chunk, "embedding": vector})
    return {
        **document, "model": provider.model_name, "dimensions": dimensions,
        "embedded_chunk_count": len(records), "chunks": records,
    }

def process_content(medium: str, content: str) -> dict:
    processor_class = PROCESSORS.get(medium)

    processor = processor_class()

    return processor.process(content)

def process_text_file(
    path: str,
    chunk_size: int,
    stride: int,
) -> dict:
    processor = TextFileProcessor(chunk_size=chunk_size, stride=stride)

    return processor.process(path)
