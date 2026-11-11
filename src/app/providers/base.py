from abc import ABC, abstractmethod

class BaseProvider(ABC):

    @abstractmethod
    def embed(self, content: list) -> list:
        pass
