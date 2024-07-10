from typing import Any

from zenbase.adaptors.base.dataset_helper import BaseDatasetHelper
from zenbase.types import LMDemo


class LunaryDatasetHelper(BaseDatasetHelper):
    def create_dataset(self, dataset_name: str, *args, **kwargs) -> Any:
        raise NotImplementedError("Lunary doesn't support creating datasets")

    def add_examples_to_dataset(self, dataset_id: Any, inputs: list, outputs: list) -> None:
        raise NotImplementedError("Lunary doesn't support adding examples to datasets")

    def fetch_dataset_examples(self, dataset_name: str):
        return self.client.get_dataset(dataset_name)

    def fetch_dataset_demos(self, dataset_name: str) -> list[LMDemo]:
        return [
            LMDemo(inputs=example.input, outputs=example.ideal_output, adaptor_object=example)
            for example in self.client.get_dataset(dataset_name)
        ]
