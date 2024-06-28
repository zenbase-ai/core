from typing import Any, Callable

import lunary

from zenbase.optim.metric.types import (
    CandidateMetricEvaluator,
    CandidateMetricResult,
    IndividualEvalMetric,
    MetricEvals,
)
from zenbase.types import LMDemo, LMFunction
from zenbase.utils import pmap


class ZenLunary:
    MetricEvaluator = Callable[[list[tuple[bool, Any]]], MetricEvals]

    @staticmethod
    def default_metric(batch_results: list[tuple[bool, Any]]) -> MetricEvals:
        avg_pass = sum(int(passed) for passed, _ in batch_results) / len(batch_results)
        return {"score": avg_pass}

    @staticmethod
    def data_to_demo(dataset_item: lunary.DatasetItem) -> LMDemo:
        return LMDemo(inputs=dataset_item.input, outputs=dataset_item.ideal_output, original_object=dataset_item)

    @classmethod
    def dataset_to_demos(cls, dataset: list[lunary.DatasetItem]) -> list[LMDemo]:
        return [cls.data_to_demo(item) for item in dataset]

    @classmethod
    def metric_evaluator(
        cls,
        *args,
        checklist: str,
        evalset: list[lunary.DatasetItem],
        eval_metrics: MetricEvaluator = default_metric,
        concurrency: int = 20,
        **kwargs,
    ) -> CandidateMetricEvaluator:
        def evaluate_metric(function: LMFunction) -> CandidateMetricResult:
            individual_evals = []

            def run_and_evaluate(demo: LMDemo):
                nonlocal individual_evals

                item = demo.original_object

                response = function(item.input)
                result = lunary.evaluate(
                    checklist,
                    input=item.input,
                    output=response,
                    ideal_output=item.ideal_output,
                    *args,
                    **kwargs,
                )

                individual_evals.append(
                    IndividualEvalMetric(
                        details=result[1][0]["details"],
                        passed=result[0],
                        response=response,
                        demo=demo,
                    )
                )

                return result

            eval_results = pmap(
                run_and_evaluate,
                cls.dataset_to_demos(evalset),
                concurrency=concurrency,
            )

            return CandidateMetricResult(function, eval_metrics(eval_results), individual_evals=individual_evals)

        return evaluate_metric
