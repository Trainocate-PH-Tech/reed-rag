import pytest

from app.processors.base import BaseProcessor
from app.processors.text import TextProcessor

@pytest.mark.unit
def test_processor_implements_interface():
    processor = TextProcessor()

    assert isinstance(processor, BaseProcessor)
    assert callable(processor.process)

@pytest.mark.unit
def test_base_processor_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseProcessor()

def test_text_processor_counts_characters_and_words(sample_text):
    processor = TextProcessor()
    result = processor.process(sample_text)

    assert result == { 
        "medium": "text", 
        "content": sample_text, 
        "characters": len(sample_text),
        "words": len(sample_text.split()),
    }
