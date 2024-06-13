from typing import Any, Callable

import lunary

from zenbase.optim.metric.types import (
    MetricEvals,
    CandidateMetricResult,
    CandidateMetricEvaluator,
)
from zenbase.types import LMDemo, LMFunction
from zenbase.utils import pmap


class ZenLunary:
    type MetricEvaluator = Callable[[list[tuple[bool, Any]]], MetricEvals]

    @staticmethod
    def default_metric(batch_results: list[tuple[bool, Any]]) -> MetricEvals:
        avg_pass = sum(int(passed) for passed, _ in batch_results) / len(batch_results)
        return {"score": avg_pass}

    @staticmethod
    def dataset_to_demos(dataset: list[lunary.DatasetItem]) -> list[LMDemo]:
        return [
            LMDemo(inputs=item.input, outputs=item.ideal_output) for item in dataset
        ]

    @classmethod
    def metric_evaluator[
        Inputs: dict, Outputs: dict
    ](
        cls,
        *args,
        checklist: str,
        evalset: list[lunary.DatasetItem],
        eval_metrics: MetricEvaluator = default_metric,
        concurrency: int = 20,
        **kwargs,
    ) -> CandidateMetricEvaluator:
        def evaluate_metric(
            function: LMFunction[Inputs, Outputs]
        ) -> CandidateMetricResult[Inputs, Outputs]:
            def run_and_evaluate(item: lunary.DatasetItem):
                response = function(item.input)
                return lunary.evaluate(
                    checklist,
                    input=item.input,
                    output=response,
                    ideal_output=item.ideal_output,
                    *args,
                    **kwargs,
                )

            eval_results = pmap(
                run_and_evaluate,
                evalset,
                concurrency=concurrency,
            )

            return CandidateMetricResult(function, eval_metrics(eval_results))

        return evaluate_metric
