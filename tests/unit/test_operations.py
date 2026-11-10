import pytest

from app.operations.process import (
    PROCESSORS,
    process_content,
)

@pytest.mark.unit
def test_processor_registry():
    assert "text" in PROCESSORS

@pytest.mark.unit
def test_process_content(sample_text):
    result = process_content(
        medium="text",
        content=sample_text,
    )

    assert result["medium"] == "text"
    assert result["content"] == sample_text
