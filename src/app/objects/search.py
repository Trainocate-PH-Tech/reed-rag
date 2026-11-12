from dataclasses import dataclass

from app.database.documents import Document

@dataclass(frozen=True)
class SearchResult:
    document: Document
    distance: float
