import pytest
from unittest.mock import Mock

from app.operations.process import (
    PROCESSORS,
    process_content,
    process_text_file,
    embed_text_file,
)

@pytest.mark.unit
def test_embed_batches(tmp_path, runtime):
    model, constructor, engine = runtime
    path = tmp_path / "protocol.txt"
    path.write_text("ABCDEFGHIJ", encoding="utf-8")
    provider = Mock(model_name="fake.gguf")
    provider.embed.side_effect = lambda texts: [[0.1, 0.2] for _ in texts]
    factory = Mock(return_value=provider)

    result = embed_text_file(
        path, 4, 3, str(model), 
        batch_size=2,
        provider_factory=factory
    )

    factory.assert_called_once_with(
        model_path=str(model), 
        context_size=2048, 
        gpu_layers=0,
    )
    assert [call.args[0] for call in provider.embed.call_args_list] == [
        ["title: none | text: ABCD", "title: none | text: DEFG"],
        ["title: none | text: GHIJ"],
    ]
    assert result["dimensions"] == 2
    assert result["model"] == "fake.gguf"
    assert result["embedded_chunk_count"] == 3
    assert result["chunks"][0] == {
        "index": 0, "start": 0, "end": 4, "text": "ABCD",
        "embedding": [0.1, 0.2],
    }

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
