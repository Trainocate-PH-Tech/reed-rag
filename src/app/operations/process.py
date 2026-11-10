from app.processors.base import BaseProcessor
from app.processors.text import TextProcessor
from app.processors.text_file import TextFileProcessor

PROCESSORS: dict[str, type[BaseProcessor]] = {
    "text": TextProcessor
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
