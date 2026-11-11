import pytest

from app.providers.local import LocalEmbeddingProvider

@pytest.mark.unit
def test_load_and_reuse_model(runtime):
    model, constructor, engine = runtime
    provider = LocalEmbeddingProvider(str(model), context_size=2048, gpu_layers=0)
    assert provider.model_name == "model.gguf"
    assert provider.embed(["first"]) == [[0.1, 0.2]]
    assert provider.embed(["second"]) == [[0.1, 0.2]]
    constructor.assert_called_once_with(
        model_path=str(model.resolve()), embedding=True,
        n_ctx=2048, n_gpu_layers=0, verbose=False,
    )
    assert engine.create_embedding.call_count == 2
    engine.create_embedding.assert_called_with(["second"])

