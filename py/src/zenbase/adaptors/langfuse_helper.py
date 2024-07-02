from typing import Callable

from langfuse import Langfuse
from langfuse.client import Dataset

from zenbase.optim.metric.types import (
    CandidateEvalResult,
    CandidateEvaluator,
    OverallEvalValue,
)
from zenbase.types import LMDemo, LMFunction, Outputs
from zenbase.utils import pmap


class ZenLangfuse:
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
        return [LMDemo(inputs=item.input, outputs=item.expected_output) for item in dataset.items]

    @classmethod
    def metric_evaluator(
        cls,
        evalset: Dataset,
        evaluate: Callable[[Outputs, LMDemo, Langfuse], OverallEvalValue],
        candidate_evals: MetricEvaluator = default_candidate_evals,
        langfuse: Langfuse | None = None,
        concurrency: int = 20,
    ) -> CandidateEvaluator:
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
