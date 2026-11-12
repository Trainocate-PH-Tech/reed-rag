from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

@dataclass
class Document:
    content: str
    embedding: list[float]
    content_type: str = ""
    id: str = field(default_factory=lambda: str(uuid4()))
    source_id: str = ""
    storage_key: str = ""
    filename: str = ""
    source: str = ""
    chunk_index: int = 0
    start: int = 0
    end: int = 0
    embedding_model: str = ""
    metadata: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def summary(self):
        values = asdict(self)
        values.pop("embedding")
        return values

def to_metadata(document):
    values = document.summary()
    values.pop("id")
    values.pop("content")
    return values


def from_record(record_id, content, embedding, metadata):
    return Document(id=record_id, content=content,
                embedding=[float(value) for value in embedding], **metadata)
