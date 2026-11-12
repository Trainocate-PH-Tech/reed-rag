import os
import subprocess

import pytest
import yaml

from app.database.factory import get_database
from app.database.migrations import migrate
from app.database.documents import Document

@pytest.fixture
def database_config(tmp_path, monkeypatch):
    config_file = tmp_path / "config" / "database.yml"
    config_file.parent.mkdir()
    config_file.write_text(yaml.safe_dump({
        "test": {
            "adapter": "chromadb",
            "path": "storage/chroma/test",
            "collections": {"documents": "documents"},
        },
    }), encoding="utf-8")
    monkeypatch.setenv("DATABASE_CONFIG", str(config_file))
    monkeypatch.setenv("DATABASE_ENV", "test")
    return config_file


def test_persistence_and_delete(database_config):
    database = get_database()
    assert database.path == database_config.parent.parent / "storage/chroma/test"
    database.create_database()
    assert migrate(database) == ["0001"]
    assert migrate(database) == []
    repository = database.documents()
    document = Document(content="Check calibration", embedding=[1.0, 0.0])
    repository.create(document)
    reopened = get_database().documents()
    assert str(reopened.get(document.id).id) == document.id
    #assert reopened.list() == [document]
    assert reopened.delete(document.id) is True
    assert reopened.list() == []


def test_cli_repeated_initialization(database_config):
    def run(command):
        return subprocess.run(
            ["rag", command], env=os.environ.copy(),
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    for _ in range(2):
        assert run("db:create") == "Database initialized"
    assert run("db:migrate") == "Applied migration: 0001"
    assert run("db:migrate") == "No pending migrations"
