import abc


class BaseLMFunctionGenerator(abc.ABC):
    @abc.abstractmethod
    def generate(self, *args, **kwargs): ...
