from abc import abstractmethod
from typing import Any

from zenbase.adaptors.base.adaptor import ZenAdaptor


class JSONDatasetHelper(ZenAdaptor):
    def create_dataset(self, dataset_name: str, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def add_examples_to_dataset(self, dataset_id: Any, inputs: list, outputs: list) -> None:
        raise NotImplementedError

    @abstractmethod
    def fetch_dataset_examples(self, dataset_name: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def fetch_dataset_demos(self, dataset: Any) -> Any:
        raise NotImplementedError
