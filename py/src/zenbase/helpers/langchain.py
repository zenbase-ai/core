from dataclasses import asdict
from typing import TYPE_CHECKING, Iterator

from zenbase.types import LMDemo, LMFunction
from zenbase.optim.metric.types import (
    MetricEvals,
    CandidateMetricResult,
    CandidateMetricEvaluator,
)
from zenbase.utils import random_name_generator

if TYPE_CHECKING:
    from langsmith import schemas


class ZenLangSmith:
    @staticmethod
    def examples_to_demos(examples: Iterator["schemas.Example"]) -> list[LMDemo]:
        return [LMDemo(inputs=e.inputs, outputs=e.outputs) for e in examples]

    @classmethod
    def metric_evaluator[
        Inputs: dict, Outputs: dict
    ](cls, **evaluate_kwargs) -> CandidateMetricEvaluator:
        from langsmith import evaluate

        metadata = evaluate_kwargs.pop("metadata", {})
        gen_random_name = random_name_generator(
            evaluate_kwargs.pop("experiment_prefix", None)
        )

        def evaluate_candidate(
            function: LMFunction[Inputs, Outputs],
        ) -> CandidateMetricResult[Inputs, Outputs]:
            experiment_results = evaluate(
                function,
                experiment_prefix=gen_random_name(),
                metadata={
                    **metadata,
                    **asdict(function.zenbase),
                },
                **evaluate_kwargs,
            )

            if summary_results := experiment_results._summary_results["results"]:
                evals = cls._eval_results_to_evals(summary_results)
            else:
                evals = cls._experiment_results_to_evals(experiment_results)

            return CandidateMetricResult(function, evals)

        return evaluate_candidate

    @staticmethod
    def _experiment_results_to_evals(experiment_results: list) -> MetricEvals:
        total = sum(
            res["evaluation_results"]["results"][0].score
            for res in experiment_results._results
        )
        count = len(experiment_results._results)
        mean = total / count
        return {"score": mean}

    @staticmethod
    def _eval_results_to_evals(eval_results: list) -> MetricEvals:
        if not eval_results:
            raise ValueError("No evaluation results")

        return {
            "score": eval_results[0].score,
            **{r.key: r.dict() for r in eval_results},
        }
