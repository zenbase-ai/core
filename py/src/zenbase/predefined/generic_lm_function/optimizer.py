from dataclasses import dataclass, field
from typing import Any, Callable, List, NamedTuple, Type

from instructor.client import Instructor
from pydantic import BaseModel

from zenbase.adaptors.json.adaptor import JSONAdaptor
from zenbase.core.managers import ZenbaseTracer
from zenbase.optim.metric.labeled_few_shot import LabeledFewShot
from zenbase.optim.metric.types import CandidateEvalResult
from zenbase.types import LMDemo, LMFunction


@dataclass
class GenericLMFunctionOptimizer:
    class Result(NamedTuple):
        best_function: LMFunction
        candidate_results: list[CandidateEvalResult]
        best_candidate_result: CandidateEvalResult | None

    instructor_client: Instructor
    prompt: str
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]
    model: str
    zenbase_tracer: ZenbaseTracer
    training_set: List[dict]
    validation_set: List[dict]
    test_set: List[dict]
    custom_evaluator: Callable[[Any, dict], dict] = field(default=None)
    shots: int = 5
    samples: int = 10
    last_result: Result | None = field(default=None)

    lm_function: LMFunction = field(init=False)
    training_set_demos: List[LMDemo] = field(init=False)
    validation_set_demos: List[LMDemo] = field(init=False)
    test_set_demos: List[LMDemo] = field(init=False)
    best_evaluation: CandidateEvalResult | None = field(default=None)
    base_evaluation: CandidateEvalResult | None = field(default=None)

    def __post_init__(self):
        self.lm_function = self._generate_lm_function()
        self.training_set_demos = self._convert_dataset_to_demos(self.training_set)
        self.validation_set_demos = self._convert_dataset_to_demos(self.validation_set)
        self.test_set_demos = self._convert_dataset_to_demos(self.test_set)

    def _generate_lm_function(self) -> LMFunction:
        @self.zenbase_tracer.trace_function
        def generic_function(request):
            system_role = "assistant" if self.model.startswith("o1") else "system"
            messages = [
                {"role": system_role, "content": self.prompt},
            ]

            if request.zenbase.task_demos:
                messages.append({"role": system_role, "content": "Here are some examples:"})
                for demo in request.zenbase.task_demos:
                    if demo.inputs == request.inputs:
                        continue
                    messages.extend(
                        [
                            {"role": "user", "content": str(demo.inputs)},
                            {"role": "assistant", "content": str(demo.outputs)},
                        ]
                    )
                messages.append({"role": system_role, "content": "Now, please answer the following question:"})

            messages.append({"role": "user", "content": str(request.inputs)})

            kwargs = {
                "model": self.model,
                "response_model": self.output_model,
                "messages": messages,
                "max_retries": 3,
            }

            if not self.model.startswith("o1"):
                kwargs.update(
                    {
                        "logprobs": True,
                        "top_logprobs": 5,
                    }
                )

            return self.instructor_client.chat.completions.create(**kwargs)

        return generic_function

    def _convert_dataset_to_demos(self, dataset: List[dict]) -> List[LMDemo]:
        return [LMDemo(inputs=item["inputs"], outputs=item["outputs"]) for item in dataset]

    def optimize(self) -> Result:
        evaluator = self.custom_evaluator or self._create_default_evaluator()
        test_evaluator = self._create_test_evaluator(evaluator)

        # Perform base evaluation
        self.base_evaluation = self._perform_base_evaluation(test_evaluator)

        optimizer = LabeledFewShot(demoset=self.training_set_demos, shots=self.shots)
        optimizer_result = optimizer.perform(
            self.lm_function,
            evaluator=JSONAdaptor.metric_evaluator(
                data=self.validation_set_demos,
                eval_function=evaluator,
            ),
            samples=self.samples,
            rounds=1,
        )

        # Evaluate best function
        self.best_evaluation = self._evaluate_best_function(test_evaluator, optimizer_result)

        self.last_result = self.Result(
            best_function=optimizer_result.best_function,
            candidate_results=optimizer_result.candidate_results,
            best_candidate_result=optimizer_result.best_candidate_result,
        )

        return self.last_result

    def _create_default_evaluator(self):
        def evaluator(output: BaseModel, ideal_output: dict) -> dict:
            return {
                "passed": int(output.model_dump(mode="json") == ideal_output),
            }

        return evaluator

    def _create_test_evaluator(self, evaluator):
        return JSONAdaptor.metric_evaluator(
            data=self.test_set_demos,
            eval_function=evaluator,
        )

    def _perform_base_evaluation(self, test_evaluator):
        """Perform the base evaluation of the LM function."""
        return test_evaluator(self.lm_function)

    def _evaluate_best_function(self, test_evaluator, optimizer_result):
        """Evaluate the best function from the optimization result."""
        return test_evaluator(optimizer_result.best_function)

    def create_lm_function_with_demos(self, prompt: str, demos: List[dict]) -> LMFunction:
        @self.zenbase_tracer.trace_function
        def lm_function_with_demos(request):
            system_role = "assistant" if self.model.startswith("o1") else "system"
            messages = [
                {"role": system_role, "content": prompt},
            ]

            # Add demos to the messages
            if demos:
                messages.append({"role": system_role, "content": "Here are some examples:"})
                for demo in demos:
                    messages.extend(
                        [
                            {"role": "user", "content": str(demo["inputs"])},
                            {"role": "assistant", "content": str(demo["outputs"])},
                        ]
                    )
                messages.append({"role": system_role, "content": "Now, please answer the following question:"})

            # Add the actual request
            messages.append({"role": "user", "content": str(request.inputs)})

            kwargs = {
                "model": self.model,
                "response_model": self.output_model,
                "messages": messages,
                "max_retries": 3,
            }

            if not self.model.startswith("o1"):
                kwargs.update(
                    {
                        "logprobs": True,
                        "top_logprobs": 5,
                    }
                )

            return self.instructor_client.chat.completions.create(**kwargs)

        return lm_function_with_demos

    def generate_csv_report(self):
        if not self.last_result:
            raise ValueError("No results to generate report from")

        best_candidate_result = self.last_result.best_candidate_result
        base_evaluation = self.base_evaluation
        best_evaluation = self.best_evaluation

        list_of_rows = [("type", "input", "ideal_output", "output", "passed", "score", "details")]

        for eval_item in best_candidate_result.individual_evals:
            list_of_rows.append(
                (
                    "best_candidate_result",
                    eval_item.demo.inputs,
                    eval_item.demo.outputs,
                    eval_item.response,
                    eval_item.passed,
                    eval_item.score,
                    eval_item.details,
                )
            )

        for eval_item in base_evaluation.individual_evals:
            list_of_rows.append(
                (
                    "base_evaluation",
                    eval_item.demo.inputs,
                    eval_item.demo.outputs,
                    eval_item.response,
                    eval_item.passed,
                    eval_item.score,
                    eval_item.details,
                )
            )

        for eval_item in best_evaluation.individual_evals:
            list_of_rows.append(
                (
                    "best_evaluation",
                    eval_item.demo.inputs,
                    eval_item.demo.outputs,
                    eval_item.response,
                    eval_item.passed,
                    eval_item.score,
                    eval_item.details,
                )
            )

        # save to csv
        import csv

        with open("report.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(list_of_rows)
