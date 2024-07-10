from typing import Callable

from langfuse import Langfuse
from langfuse.client import Dataset

from zenbase.adaptors.base.evaluation_helper import BaseEvaluationHelper
from zenbase.optim.metric.types import (
    CandidateEvalResult,
    CandidateEvaluator,
    IndividualEvalValue,
    OverallEvalValue,
)
from zenbase.types import LMDemo, LMFunction, Outputs
from zenbase.utils import pmap


class LangfuseEvaluationHelper(BaseEvaluationHelper):
    MetricEvaluator = Callable[[list[OverallEvalValue]], OverallEvalValue]

    @staticmethod
    def default_candidate_evals(item_evals: list[OverallEvalValue]) -> OverallEvalValue:
        keys = item_evals[0].keys()
        evals = {k: sum(d[k] for d in item_evals) / len(item_evals) for k in keys}
        if not evals["score"]:
            evals["score"] = sum(evals.values()) / len(evals)
        return evals

    @staticmethod
    def dataset_demos(dataset: Dataset) -> list[LMDemo]:
        # TODO: Should remove and deprecate
        return [LMDemo(inputs=item.input, outputs=item.expected_output) for item in dataset.items]

    @staticmethod
    def _metric_evaluator_generator(
        data: list[LMDemo],
        evaluate: Callable[[Outputs, LMDemo, Langfuse], OverallEvalValue],
        candidate_evals: MetricEvaluator = default_candidate_evals,
        langfuse: Langfuse | None = None,
        concurrency: int = 20,
        threshold: float = 0.5,
    ) -> CandidateEvaluator:
        # TODO: this is not the way to run experiment in the langfuse, we should update with the new beta feature
        from langfuse import Langfuse
        from langfuse.decorators import observe

        langfuse = langfuse or Langfuse()

        def evaluate_candidate(function: LMFunction) -> CandidateEvalResult:
            individual_evals = []

            @observe()
            def run_and_evaluate(demo: LMDemo):
                nonlocal individual_evals

                outputs = function(demo.inputs)
                evals = evaluate(outputs, demo, langfuse=langfuse)
                individual_evals.append(
                    IndividualEvalValue(
                        passed=evals["score"] >= threshold,
                        response=outputs,
                        demo=demo,
                        score=evals["score"],
                    )
                )
                return evals

            item_evals = pmap(
                run_and_evaluate,
                data,
                concurrency=concurrency,
            )
            candidate_eval = candidate_evals(item_evals)

            return CandidateEvalResult(function, candidate_eval, individual_evals=individual_evals)

        return evaluate_candidate

    @classmethod
    def metric_evaluator(
        cls,
        evalset: Dataset,
        evaluate: Callable[[Outputs, LMDemo, Langfuse], OverallEvalValue],
        candidate_evals: MetricEvaluator = default_candidate_evals,
        langfuse: Langfuse | None = None,
        concurrency: int = 20,
        threshold: float = 0.5,
    ) -> CandidateEvaluator:
        # TODO: Should remove and deprecate
        from langfuse import Langfuse
        from langfuse.decorators import observe

        langfuse = langfuse or Langfuse()

        def evaluate_candidate(function: LMFunction) -> CandidateEvalResult:
            @observe()
            def run_and_evaluate(demo: LMDemo):
                outputs = function(demo.inputs)
                evals = evaluate(outputs, demo, langfuse=langfuse)
                return evals

            item_evals = pmap(
                run_and_evaluate,
                cls.dataset_demos(evalset),
                concurrency=concurrency,
            )
            candidate_eval = candidate_evals(item_evals)

            return CandidateEvalResult(function, candidate_eval)

        return evaluate_candidate

    def get_evaluator(self, data: str):
        raise NotImplementedError("This method should be implemented by the parent class as it needs access to data")
