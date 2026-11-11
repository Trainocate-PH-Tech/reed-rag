import pytest

from app.providers.base import BaseProvider
from app.providers.local import LocalEmbeddingProvider

@pytest.mark.unit
def test_local_embedding_provider_interface(runtime):
    model, constructor, engine = runtime
    provider = LocalEmbeddingProvider(str(model))

    assert isinstance(provider, BaseProvider)
    assert callable(provider.embed)
    assert provider.model_name == "model.gguf"
