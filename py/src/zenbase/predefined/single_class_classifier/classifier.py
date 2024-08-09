__all__ = ["SingleClassClassifier"]

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, NamedTuple, Type

import cloudpickle
from instructor.client import AsyncInstructor, Instructor
from pydantic import BaseModel

from zenbase.adaptors.json.adaptor import JSONAdaptor
from zenbase.core.managers import ZenbaseTracer
from zenbase.optim.metric.labeled_few_shot import LabeledFewShot
from zenbase.optim.metric.types import CandidateEvalResult
from zenbase.predefined.base.optimizer import BasePredefinedOptimizer
from zenbase.predefined.single_class_classifier.function_generator import SingleClassClassifierLMFunctionGenerator
from zenbase.predefined.syntethic_data.single_class_classifier import SingleClassClassifierSyntheticDataExample
from zenbase.types import Inputs, LMDemo, LMFunction, Outputs


@dataclass(kw_only=True)
class SingleClassClassifier(BasePredefinedOptimizer):
    """
    A single-class classifier that optimizes and evaluates language model functions.
    """

    class Result(NamedTuple):
        best_function: LMFunction[Inputs, Outputs]
        candidate_results: list[CandidateEvalResult]
        best_candidate_result: CandidateEvalResult | None

    instructor_client: Instructor | AsyncInstructor
    prompt: str
    class_dict: Dict[str, str] | None = field(default=None)
    class_enum: Enum | None = field(default=None)
    prediction_class: Type[BaseModel] | None = field(default=None)
    model: str
    zenbase_tracer: ZenbaseTracer
    lm_function: LMFunction | None = field(default=None)
    training_set: list
    test_set: list
    validation_set: list
    shots: int = 5
    samples: int = 10
    best_evaluation: CandidateEvalResult | None = field(default=None)
    base_evaluation: CandidateEvalResult | None = field(default=None)
    optimizer_result: Result | None = field(default=None)

    def __post_init__(self):
        """Initialize the SingleClassClassifier after creation."""
        self.lm_function = self._generate_lm_function()
        self.training_set_demos = self._convert_dataset_to_demos(self.training_set)
        self.test_set_demos = self._convert_dataset_to_demos(self.test_set)
        self.validation_set_demos = self._convert_dataset_to_demos(self.validation_set)

    def _generate_lm_function(self) -> LMFunction:
        """Generate the language model function."""
        return SingleClassClassifierLMFunctionGenerator(
            instructor_client=self.instructor_client,
            prompt=self.prompt,
            class_dict=self.class_dict,
            class_enum=self.class_enum,
            prediction_class=self.prediction_class,
            model=self.model,
            zenbase_tracer=self.zenbase_tracer,
        ).generate()

    @staticmethod
    def _convert_dataset_to_demos(dataset: list) -> list[LMDemo]:
        """Convert a dataset to a list of LMDemo objects."""
        if dataset:
            if isinstance(dataset[0], dict):
                return [
                    LMDemo(inputs={"question": item["inputs"]}, outputs={"answer": item["outputs"]}) for item in dataset
                ]
            elif isinstance(dataset[0], SingleClassClassifierSyntheticDataExample):
                return [LMDemo(inputs={"question": item.inputs}, outputs={"answer": item.outputs}) for item in dataset]

    def load_classifier(self, filename: str):
        with open(filename, "rb") as f:
            lm_zenbase = cloudpickle.load(f)
            return self.lm_function.clean_and_duplicate(lm_zenbase)

    def optimize(self) -> Result:
        """
        Perform the optimization and evaluation of the language model function.

        Returns:
            Result: The optimization result containing the best function and evaluation metrics.
        """
        # Define the evaluation function
        evaluator = self._create_evaluator()

        # Create test evaluator
        test_evaluator = self._create_test_evaluator(evaluator)

        # Perform base evaluation
        self.base_evaluation = self._perform_base_evaluation(test_evaluator)

        # Create and run optimizer
        optimizer_result = self._run_optimization(evaluator)

        # Evaluate best function
        self.best_evaluation = self._evaluate_best_function(test_evaluator, optimizer_result)

        # Save last optimizer_result
        self.optimizer_result = optimizer_result

        return optimizer_result

    @staticmethod
    def _create_evaluator():
        """Create the evaluation function."""

        def evaluator(output: Any, ideal_output: Dict[str, Any]) -> Dict[str, int]:
            return {
                "passed": int(ideal_output["answer"] == output.class_label.name),
            }

        return evaluator

    def _create_test_evaluator(self, evaluator):
        """Create the test evaluator using JSONAdaptor."""
        return JSONAdaptor.metric_evaluator(
            data=self.validation_set_demos,
            eval_function=evaluator,
        )

    def _perform_base_evaluation(self, test_evaluator):
        """Perform the base evaluation of the LM function."""
        return test_evaluator(self.lm_function)

    def _run_optimization(self, evaluator):
        """Run the optimization process."""
        optimizer = LabeledFewShot(demoset=self.training_set_demos, shots=self.shots)
        return optimizer.perform(
            self.lm_function,
            evaluator=JSONAdaptor.metric_evaluator(
                data=self.validation_set_demos,
                eval_function=evaluator,
            ),
            samples=self.samples,
            rounds=1,
        )

    def _evaluate_best_function(self, test_evaluator, optimizer_result):
        """Evaluate the best function from the optimization result."""
        return test_evaluator(optimizer_result.best_function)
