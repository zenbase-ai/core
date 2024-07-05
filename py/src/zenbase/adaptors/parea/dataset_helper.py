import json
from typing import ValuesView

from parea.schemas import TestCase, TestCaseCollection

from zenbase.adaptors.base.dataset_helper import BaseDatasetHelper
from zenbase.types import LMDemo


class PareaDatasetHelper(BaseDatasetHelper):
    def create_dataset(self, dataset_name: str):
        dataset = self.client.create_test_collection(data=[], name=dataset_name)
        return dataset

    def add_examples_to_dataset(self, inputs, outputs, dataset_name: str):
        data = [{"inputs": inputs[i], "target": outputs[i]} for i in range(len(inputs))]
        self.client.add_test_cases(data, dataset_name)

    def create_dataset_and_add_examples(self, inputs, outputs, dataset_name: str):
        data = [{"inputs": inputs[i], "target": outputs[i]} for i in range(len(inputs))]
        dataset = self.client.create_test_collection(dataset_name, data)
        return dataset

    def fetch_dataset_examples(self, dataset_name: str):
        return self.fetch_dataset(dataset_name).test_cases.values()

    def fetch_dataset(self, dataset_name: str) -> TestCaseCollection:
        return self.client.get_collection(dataset_name)

    def fetch_dataset_demos(self, dataset_name: str) -> list[LMDemo]:
        return self.example_to_demo(self.fetch_dataset_examples(dataset_name))

    def fetch_dataset_list_of_dicts(self, dataset_name: str) -> list[dict]:
        return [
            {"inputs": json.loads(case.inputs["inputs"]), "target": case.target}
            for case in self.fetch_dataset_examples(dataset_name)
        ]

    @staticmethod
    def example_to_demo(examples: ValuesView[TestCase]) -> list[LMDemo]:
        return [LMDemo(inputs=example.inputs, outputs={"target": example.target}) for example in examples]
