from pathlib import Path

from app.inference.base import InferenceProvider

class LocalInferenceProvider(InferenceProvider):
    def __init__(self, model_path: str):
        path = Path(model_path).expanduser().resolve()

        from llama_cpp import Llama

        self._model_name = path.name
        self._model = Llama(
            model_path=str(path), n_ctx=4096, n_gpu_layers=0, verbose=False,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def infer(self, query: str) -> str:
        query = query.strip()

        response = self._model.create_chat_completion(
            messages=[{"role": "user", "content": query}],
            max_tokens=1024, temperature=0.7, stream=False,
        )

        answer = response["choices"][0]["message"]["content"]

        return answer.strip()

    def close(self) -> None:
        self._model.close()
