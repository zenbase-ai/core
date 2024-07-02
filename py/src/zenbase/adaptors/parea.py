import json
from dataclasses import asdict
from typing import Callable

from parea import Parea
from parea.schemas import ExperimentStatsSchema, TestCaseCollection

from zenbase.optim.metric.types import (
    CandidateEvalResult,
    CandidateEvaluator,
    OverallEvalValue,
)
from zenbase.types import LMDemo, LMFunction
from zenbase.utils import random_name_generator


class ZenParea:
    MetricEvaluator = Callable[[dict[str, float]], OverallEvalValue]

    @staticmethod
    def collection_demos(collection: TestCaseCollection) -> list[LMDemo]:
        return [LMDemo(inputs=case.inputs, outputs={"target": case.target}) for case in collection.test_cases.values()]

    @staticmethod
    def default_candidate_evals(stats: ExperimentStatsSchema) -> OverallEvalValue:
        return {**stats.avg_scores, "score": sum(stats.avg_scores.values())}

    @classmethod
    def metric_evaluator(
        cls,
        *args,
        p: Parea | None = None,
        candidate_evals: MetricEvaluator = default_candidate_evals,
        **kwargs,
    ) -> CandidateEvaluator:
        p = p or Parea()
        assert isinstance(p, Parea)

        base_metadata = kwargs.pop("metadata", {})
        gen_random_name = random_name_generator(kwargs.pop("name", None))

        def evaluate_candidate(function: LMFunction) -> CandidateEvalResult:
            experiment = p.experiment(
                func=function,
                *args,
                **kwargs,
                name=gen_random_name(),
                metadata={
                    **base_metadata,
                    "zenbase": json.dumps(asdict(function.zenbase)),
                },
            )

            experiment.run()
            assert experiment.experiment_stats, "failed to run experiment"

            return CandidateEvalResult(
                function,
                evals=candidate_evals(experiment.experiment_stats),
            )

        return evaluate_candidate
