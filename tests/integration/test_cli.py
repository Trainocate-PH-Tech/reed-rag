import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from protocol_embeddings.cli import main


@pytest.mark.parametrize("write_file", [False, True])
def test_cli_generates_embeddings(tmp_path, monkeypatch, capsys, write_file):
    protocol = tmp_path / "protocol.txt"
    protocol.write_text("ABCDEFGHIJ", encoding="utf-8")
    model = tmp_path / "model.gguf"
    model.touch()
    engine = Mock()
    engine.create_embedding.side_effect = lambda texts: {"data": [
        {"index": i, "embedding": [0.1, 0.2]} for i, _ in enumerate(texts)
    ]}
    constructor = Mock(return_value=engine)
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=constructor))

    # rag embed /home/ralampay/Documents/sample.txt --model gemma.gguf --chunk_size 4 --stride 3 --batch_size 2
    args = ["embed", str(protocol), "--model", str(model),
            "--chunk_size", "4", "--stride", "3", "--batch_size", "2"]
    output = tmp_path / "output.json"
    if write_file:
        args += ["--output", str(output)]

    main(args)

    stdout = capsys.readouterr().out
    result = json.loads(output.read_text() if write_file else stdout)
    assert result["source"] == str(protocol)
    assert result["dimensions"] == 2
    assert result["embedded_chunk_count"] == 3
    assert [chunk["text"] for chunk in result["chunks"]] == ["ABCD", "DEFG", "GHIJ"]
    assert result["chunks"][1]["start"] == 3
    assert result["chunks"][0]["embedding"] == [0.1, 0.2]
    constructor.assert_called_once()
    assert engine.create_embedding.call_args_list[0].args[0] == [
        "title: none | text: ABCD", "title: none | text: DEFG",
    ]
    if write_file:
        assert stdout == ""
