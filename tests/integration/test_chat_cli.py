import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from rag.cli import main

@pytest.fixture
def chat_model(tmp_path, monkeypatch):
    path = tmp_path / "chat.gguf"
    path.touch()
    engine = Mock()
    engine.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Hello!"}}]
    }
    constructor = Mock(return_value=engine)
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=constructor))
    return path, engine, constructor

@pytest.mark.integration
def test_single_query(chat_model, capsys):
    path, engine, constructor = chat_model
    main(["chat", "--model", str(path), "--query", "  Hello  "])

    assert capsys.readouterr().out == "Hello!\n"
    constructor.assert_called_once_with(
        model_path=str(path.resolve()), n_ctx=4096,
        n_gpu_layers=0, verbose=False,
    )
    engine.create_chat_completion.assert_called_once_with(
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=512, temperature=0.7, stream=False,
    )
    engine.close.assert_called_once()
