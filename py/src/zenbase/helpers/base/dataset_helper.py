from abc import abstractmethod
from typing import Any

from zenbase.helpers.base.adaptor import ZenAdaptor


class BaseDatasetHelper(ZenAdaptor):
    @abstractmethod
    def create_dataset(self, dataset_name: str, description: str) -> Any: ...

    @abstractmethod
    def add_examples_to_dataset(self, dataset_id: Any, inputs: list, outputs: list) -> None: ...

    @abstractmethod
    def fetch_dataset_examples(self, dataset_name: str) -> Any: ...

    @abstractmethod
    def fetch_dataset_demos(self, dataset: Any) -> Any: ...
