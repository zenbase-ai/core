from typing import Any

from zenbase.adaptors.base.adaptor import ZenAdaptor
from zenbase.types import LMDemo


class JSONDatasetHelper(ZenAdaptor):
    datasets = {}

    def create_dataset(self, dataset_name: str, *args, **kwargs) -> Any:
        self.datasets[dataset_name] = []

        return self.datasets[dataset_name]

    def add_examples_to_dataset(self, dataset_id: Any, inputs: list, outputs: list) -> None:
        for input, output in zip(inputs, outputs):
            self.datasets[dataset_id].append({"input": input, "output": output})

    def fetch_dataset_examples(self, dataset_name: str) -> Any:
        return self.datasets[dataset_name]

    def fetch_dataset_demos(self, dataset: Any) -> Any:
        if isinstance(dataset[0], LMDemo):
            return dataset
        return [LMDemo(input=item, output=item) for item in dataset]
