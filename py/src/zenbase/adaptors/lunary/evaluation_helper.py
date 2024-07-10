from typing import Any, Callable

import lunary

from zenbase.adaptors.base.evaluation_helper import BaseEvaluationHelper
from zenbase.optim.metric.types import (
    CandidateEvalResult,
    CandidateEvaluator,
    IndividualEvalValue,
    OverallEvalValue,
)
from zenbase.types import LMDemo, LMFunction
from zenbase.utils import pmap


class LunaryEvaluationHelper(BaseEvaluationHelper):
    MetricEvaluator = Callable[[list[tuple[bool, Any]]], OverallEvalValue]

    @staticmethod
    def default_metric(batch_results: list[tuple[bool, Any]]) -> OverallEvalValue:
        avg_pass = sum(int(passed) for passed, _ in batch_results) / len(batch_results)
        return {"score": avg_pass}

    def get_evaluator(self, data: str):
        raise NotImplementedError("This method should be implemented by the parent class as it needs access to data")

    @staticmethod
    def _metric_evaluator_generator(
        *args,
        checklist: str,
        data: list[LMDemo],
        eval_metrics: MetricEvaluator = default_metric,
        concurrency: int = 1,
        **kwargs,
    ) -> CandidateEvaluator:
        # TODO: Should remove and deprecate
        def evaluate_metric(function: LMFunction) -> CandidateEvalResult:
            individual_evals = []

            def run_and_evaluate(demo: LMDemo):
                nonlocal individual_evals

                response = function(demo.inputs)
                result = lunary.evaluate(
                    checklist,
                    input=demo.inputs,
                    output=response,
                    ideal_output=demo.outputs,
                    *args,
                    **kwargs,
                )

                individual_evals.append(
                    IndividualEvalValue(
                        details=result[1][0]["details"],
                        passed=result[0],
                        response=response,
                        demo=demo,
                    )
                )

                return result

            eval_results = pmap(
                run_and_evaluate,
                data,
                concurrency=concurrency,
            )

            return CandidateEvalResult(function, eval_metrics(eval_results), individual_evals=individual_evals)

        return evaluate_metric

    @classmethod
    def dataset_to_demos(cls, dataset: list[lunary.DatasetItem]) -> list[LMDemo]:
        # TODO: Should remove and deprecate
        return [LMDemo(inputs=item.input, outputs=item.ideal_output, adaptor_object=item) for item in dataset]

    @classmethod
    def metric_evaluator(
        cls,
        *args,
        checklist: str,
        evalset: list[lunary.DatasetItem],
        eval_metrics: MetricEvaluator = default_metric,
        concurrency: int = 20,
        **kwargs,
    ) -> CandidateEvaluator:
        # TODO: Should remove and deprecate
        def evaluate_metric(function: LMFunction) -> CandidateEvalResult:
            individual_evals = []

            def run_and_evaluate(demo: LMDemo):
                nonlocal individual_evals

                item = demo.adaptor_object

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
                    IndividualEvalValue(
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

            return CandidateEvalResult(function, eval_metrics(eval_results), individual_evals=individual_evals)

        return evaluate_metric
