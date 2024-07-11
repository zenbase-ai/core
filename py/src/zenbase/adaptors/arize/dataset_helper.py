import pandas as pd
from phoenix.experiments.types import Dataset, Example

from zenbase.adaptors.base.dataset_helper import BaseDatasetHelper
from zenbase.types import LMDemo


class ArizeDatasetHelper(BaseDatasetHelper):
    def create_dataset(self, dataset_name: str, *args, **kwargs):
        raise NotImplementedError(
            "create_dataset not implemented / supported for Arize, dataset will be created"
            "automatically when adding examples to it."
        )

    def add_examples_to_dataset(self, dataset_name: str, inputs: list, outputs: list) -> Dataset:
        list_of_examples = []
        for inputs, outputs in zip(inputs, outputs):
            list_of_examples.append({"inputs": inputs, "outputs": outputs})
        df = pd.DataFrame(list_of_examples)
        return self.client.upload_dataset(
            dataset_name=dataset_name,
            dataframe=df,
            input_keys=["inputs"],
            output_keys=["outputs"],
        )

    def fetch_dataset_examples(self, dataset_name: str) -> list[Example]:
        return list(self.client.get_dataset(name=dataset_name).examples.values())

    def fetch_dataset_demos(self, dataset_name: str) -> list[LMDemo]:
        dataset_examples = self.fetch_dataset_examples(dataset_name)
        return [
            LMDemo(inputs=example.input, outputs=example.output)  # noqa
            for example in dataset_examples
        ]
