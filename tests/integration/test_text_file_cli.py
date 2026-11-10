import subprocess
import sys
import pytest
import json

def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "rag.cli", *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )

@pytest.mark.integration
def test_cli_process_text_file(sample_text_file):
    result = run_cli(
        "process",
        "text-file",
        sample_text_file,
        "--chunk_size=4",
        "--stride=3",
    )

    assert result.returncode == 0, result.stderr

    output = json.loads(result.stdout)

    assert output["medium"] == "text-file"
    assert output["chunk_size"] == 4
    assert output["stride"] == 3
    assert output["total_characters"] == 10
    assert output["chunk_content"] == 3
