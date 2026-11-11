from abc import ABC, abstractmethod

class InferenceProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def infer(self, query: str) -> str:
        """Answer one query without conversation history."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
