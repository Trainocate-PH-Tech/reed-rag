from app.processors.base import BaseProcessor
from pathlib import Path

class TextFileProcessor(BaseProcessor):
    def __init__(self, chunk_size: int, stride: int):
        self.chunk_size = chunk_size
        self.stride = stride

    def process(self, content: str) -> dict:
        path = Path(content)

        text = path.read_text(encoding="utf-8")

        chunks = []

        for start in range(0, len(text), self.stride):
            end = min(start + self.chunk_size, len(text))

            chunks.append({
                "index": len(chunks),
                "start": start,
                "end": end,
                "text": text[start:end],
            })

            if end == len(text):
                break

        return {
            "medium": "text-file",
            "chunk_size": self.chunk_size,
            "stride": self.stride,
            "total_characters": len(text),
            "chunk_content": len(chunks),
            "chunks": chunks
        }
