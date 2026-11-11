from app.providers.base import BaseProvider
from pathlib import Path
import math

class LocalEmbeddingProvider(BaseProvider):
    def __init__(self, model_name, context_size=2048, gpu_layers=0):
       
        path = Path(model_name).expanduser().resolve()
        self.model_name = path.name
        self.context_size = context_size
        self.gpu_layers = gpu_layers

        from llama_cpp import Llama

        self._model = Llama(
            model_path=str(path), 
            embedding=True,
            n_ctx=context_size, 
            n_gpu_layers=gpu_layers, 
            verbose=False,
        )

    def embed(self, texts: list) -> list:
        if not texts:
            return []
        response = self._model.create_embedding(texts)
        rows = sorted(response["data"], key=lambda row: row["index"])
        if len(rows) != len(texts):
            raise ValueError("Embedding count mismatch")

        vectors = []
        for index, row in enumerate(rows):
            if row["index"] != index:
                raise ValueError("Missing or duplicate embedding index")
            vector = row["embedding"]
            if not isinstance(vector, list) or not vector:
                raise ValueError("Expected a nonempty vector")
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in vector
            ):
                raise ValueError("Invalid embedding value")
            if vectors and len(vector) != len(vectors[0]):
                raise ValueError("Inconsistent vector dimensions")
            vectors.append(vector)
        return vectors
