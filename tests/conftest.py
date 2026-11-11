import pytest
from types import SimpleNamespace
import sys
from unittest.mock import Mock

@pytest.fixture
def runtime(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()

    engine = Mock()

    engine.create_embedding.return_value = {
        "data": [{"index": 0, "embedding": [0.1, 0.2]}]
    }

    constructor = Mock(return_value=engine)
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=constructor))
    return model, constructor, engine

@pytest.fixture
def sample_text():
    return "Hello world!"

@pytest.fixture
def empty_text():
    return ""

@pytest.fixture
def sample_text_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("ABCDEFGHIJ", encoding="utf-8")

    return file_path
