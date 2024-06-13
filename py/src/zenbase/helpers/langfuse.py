from typing import Callable

from langfuse import Langfuse
from langfuse.client import Dataset

from zenbase.optim.metric.types import (
    MetricEvals,
    CandidateMetricResult,
    CandidateMetricEvaluator,
)
from zenbase.types import LMDemo, LMFunction
from zenbase.utils import pmap


class ZenLangfuse:
    type MetricEvaluator = Callable[[list[MetricEvals]], MetricEvals]

    @staticmethod
    def default_candidate_evals(item_evals: list[MetricEvals]) -> MetricEvals:
        keys = item_evals[0].keys()
        evals = {k: sum(d[k] for d in item_evals) / len(item_evals) for k in keys}
        if not evals["score"]:
            evals["score"] = sum(evals.values()) / len(evals)
        return evals

    @staticmethod
    def dataset_demos(dataset: Dataset) -> list[LMDemo]:
        return [
            LMDemo(inputs=item.input, outputs=item.expected_output)
            for item in dataset.items
        ]

    @classmethod
    def metric_evaluator[
        Inputs: dict, Outputs: dict
    ](
        cls,
        evalset: Dataset,
        evaluate: Callable[[Outputs, LMDemo, Langfuse], MetricEvals],
        candidate_evals: MetricEvaluator = default_candidate_evals,
        langfuse: Langfuse | None = None,
        concurrency: int = 20,
    ) -> CandidateMetricEvaluator:
        from langfuse import Langfuse
        from langfuse.decorators import observe

        langfuse = langfuse or Langfuse()

        def evaluate_candidate(
            function: LMFunction[Inputs, Outputs],
        ) -> CandidateMetricResult[Inputs, Outputs]:
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

            return CandidateMetricResult(function, candidate_eval)

        return evaluate_candidate
