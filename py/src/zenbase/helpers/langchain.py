from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, Iterator

from langsmith import Client

from zenbase.optim.metric.types import (
    CandidateEvalResult,
    CandidateEvaluator,
    IndividualEvalValue,
    OverallEvalValue,
)
from zenbase.types import LMDemo, LMFunction
from zenbase.utils import random_name_generator

if TYPE_CHECKING:
    from langsmith import schemas


class ZenLangSmith:
    def __init__(self, client):
        self.client = client if client else Client()

    def create_dataset(self, dataset_name: str, description: str) -> "schemas.Dataset":
        # Create a new dataset in LangSmith
        dataset = self.client.create_dataset(dataset_name, description=description)
        return dataset

    def add_examples_to_dataset(self, dataset_id: str, inputs: list, outputs: list) -> None:
        """
        Add examples to the dataset in LangSmith.

        Parameters:
        dataset_id (str): The ID of the dataset.
        inputs (list): A list of input examples.
        outputs (list): A list of output examples.
        """
        # Create examples in LangSmith
        self.client.create_examples(
            inputs=inputs,
            outputs=outputs,
            dataset_id=dataset_id,
        )

    def fetch_dataset(self, dataset_name: str) -> Dict[str, Any]:
        # Fetch the dataset by name
        datasets = self.client.list_datasets(dataset_name=dataset_name)
        if not datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        dataset = [i for i in datasets][0]  # Assume the first match is the one we want

        # Fetch all examples for this dataset
        examples = list(self.client.list_examples(dataset_id=dataset.id))

        # Prepare the return dictionary
        dataset_dict = {
            "dataset_info": {
                "id": dataset.id,
                "name": dataset.name,
                "description": dataset.description,
                "created_at": str(dataset.created_at),
            },
            "examples": [],
        }

        # Add examples to the dictionary
        for example in examples:
            dataset_dict["examples"].append(
                {
                    "id": example.id,
                    "inputs": example.inputs,
                    "outputs": example.outputs,
                    "created_at": str(example.created_at),
                }
            )

        return dataset_dict

    def fetch_dataset_demos(self, dataset_name: str) -> list[LMDemo]:
        dataset_dict = self.fetch_dataset(dataset_name)
        return [LMDemo(inputs=example["inputs"], outputs=example["outputs"]) for example in dataset_dict["examples"]]

    @staticmethod
    def examples_to_demos(examples: Iterator["schemas.Example"]) -> list[LMDemo]:
        return [LMDemo(inputs=e.inputs, outputs=e.outputs) for e in examples]

    @classmethod
    def metric_evaluator(cls, threshold=0.5, **evaluate_kwargs) -> CandidateEvaluator:
        from langsmith import evaluate

        metadata = evaluate_kwargs.pop("metadata", {})
        gen_random_name = random_name_generator(evaluate_kwargs.pop("experiment_prefix", None))

        def evaluate_candidate(function: LMFunction) -> CandidateEvalResult:
            experiment_results = evaluate(
                function,
                experiment_prefix=gen_random_name(),
                metadata={
                    **metadata,
                    **asdict(function.zenbase),
                },
                **evaluate_kwargs,
            )

            individual_evals = cls._experiment_results_to_individual_evals(experiment_results, threshold)

            if summary_results := experiment_results._summary_results["results"]:
                # this is the case that user setup summary evaluator in the langsmith evaluate function
                evals = cls._eval_results_to_evals(summary_results)
            else:
                # this is the case that user did not setup summary evaluator in the langsmith evaluate function
                evals = cls._individual_evals_to_overall_evals(individual_evals)

            return CandidateEvalResult(function, evals, individual_evals)

        return evaluate_candidate

    @staticmethod
    def _individual_evals_to_overall_evals(individual_evals: list[IndividualEvalValue]) -> OverallEvalValue:
        """
        if the score is defined, let's get the average of the score, if not let's use the pass and fail rate
        """
        if not individual_evals:
            raise ValueError("No evaluation results")

        if individual_evals[0].score is not None:
            number_of_scores = sum(1 for e in individual_evals if e.score is not None)
            score = sum(e.score for e in individual_evals if e.score is not None) / number_of_scores
        else:
            number_of_filled_passed = sum(1 for e in individual_evals if e.passed is not None)
            number_of_actual_passed = sum(1 for e in individual_evals if e.passed)
            score = number_of_actual_passed / number_of_filled_passed

        return {"score": score}

    @staticmethod
    def _experiment_results_to_individual_evals(experiment_results: list, threshold=0.5) -> list[IndividualEvalValue]:
        individual_evals = []
        for res in experiment_results._results:
            score = res["evaluation_results"]["results"][0].score
            inputs = res["example"].inputs
            outputs = res["example"].outputs
            individual_evals.append(
                IndividualEvalValue(
                    passed=score >= threshold,
                    response=outputs,
                    demo=LMDemo(inputs=inputs, outputs=outputs, original_object=res["example"]),
                    details=res,
                    score=score,
                )
            )
        return individual_evals

    @staticmethod
    def _eval_results_to_evals(eval_results: list) -> OverallEvalValue:
        if not eval_results:
            raise ValueError("No evaluation results")

        return {
            "score": eval_results[0].score,
            **{r.key: r.dict() for r in eval_results},
        }
