from typing import Any

from langfuse.client import DatasetItemClient

from zenbase.adaptors.base.dataset_helper import BaseDatasetHelper
from zenbase.types import LMDemo


class LangfuseDatasetHelper(BaseDatasetHelper):
    def create_dataset(self, dataset_name: str, *args, **kwargs) -> Any:
        return self.client.create_dataset(dataset_name, *args, **kwargs)

    def add_examples_to_dataset(self, dataset_name: str, inputs: list, outputs: list) -> None:
        for the_input, the_output in zip(inputs, outputs):
            self.client.create_dataset_item(
                dataset_name=dataset_name,
                input=the_input,
                expected_output=the_output,
            )

    def fetch_dataset_examples(self, dataset_name: str) -> list[DatasetItemClient]:
        return self.client.get_dataset(dataset_name).items

    def fetch_dataset_demos(self, dataset_name: str) -> list[LMDemo]:
        return [
            LMDemo(inputs=example.input, outputs=example.expected_output)
            for example in self.fetch_dataset_examples(dataset_name)
        ]
