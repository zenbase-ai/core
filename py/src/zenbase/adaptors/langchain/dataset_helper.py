from typing import TYPE_CHECKING, Iterator

from zenbase.adaptors.base.dataset_helper import BaseDatasetHelper
from zenbase.types import LMDemo

if TYPE_CHECKING:
    from langsmith import schemas


class LangsmithDatasetHelper(BaseDatasetHelper):
    def create_dataset(self, dataset_name: str, description: str) -> "schemas.Dataset":
        dataset = self.client.create_dataset(dataset_name, description=description)
        return dataset

    def add_examples_to_dataset(self, dataset_name: str, inputs: list, outputs: list) -> None:
        self.client.create_examples(
            inputs=inputs,
            outputs=outputs,
            dataset_name=dataset_name,
        )

    def fetch_dataset_examples(self, dataset_name: str) -> Iterator["schemas.Example"]:
        dataset = self.fetch_dataset(dataset_name)
        return self.client.list_examples(dataset_id=dataset.id)

    def fetch_dataset(self, dataset_name: str):
        datasets = self.client.list_datasets(dataset_name=dataset_name)
        if not datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        dataset = [i for i in datasets][0]
        return dataset

    def fetch_dataset_demos(self, dataset: "schemas.Dataset") -> list[LMDemo]:
        dataset_examples = self.fetch_dataset_examples(dataset.name)
        return self.examples_to_demos(dataset_examples)

    @staticmethod
    def examples_to_demos(examples: Iterator["schemas.Example"]) -> list[LMDemo]:
        return [LMDemo(inputs=e.inputs, outputs=e.outputs, adaptor_object=e) for e in examples]
