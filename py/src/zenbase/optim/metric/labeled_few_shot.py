from dataclasses import dataclass, field
from math import factorial
from typing import NamedTuple

from zenbase.optim.abc import LMOptim
from zenbase.optim.metric.types import CandidateMetricEvaluator, CandidateMetricResult
from zenbase.types import LMDemo, LMFunction, LMZenbase
from zenbase.utils import asyncify, pmap, tracer, get_logger


log = get_logger(__name__)


@dataclass(kw_only=True)
class LabeledFewShot[Inputs: dict, Outputs: dict](LMOptim):
    class Result(NamedTuple):
        best_function: LMFunction[Inputs, Outputs]
        candidate_results: list[CandidateMetricResult]

    demoset: list[LMDemo[Inputs, Outputs]]
    shots: int = field(default=5)

    def __post_init__(self):
        assert 1 <= self.shots <= len(self.demoset)

    @tracer.start_as_current_span("perform")
    def perform(
        self,
        lmfn: LMFunction[Inputs, Outputs],
        evaluator: CandidateMetricEvaluator[Inputs, Outputs],
        deps: list[LMFunction[Inputs, Outputs]] = [],
        samples: int = 0,
        rounds: int = 1,
        concurrency: int = 1,
    ) -> Result:
        samples = samples or len(self.demoset)

        best_score = float("-inf")
        best_lmfn = lmfn

        @tracer.start_as_current_span("run_experiment")
        def run_candidate_zenbase(zenbase: LMZenbase):
            nonlocal best_score, best_lmfn

            candidate_fn = lmfn.refine(zenbase)
            candidate_result = evaluator(candidate_fn)

            self.events.emit("candidate", candidate_result)

            if candidate_result.evals["score"] > best_score:
                best_score = candidate_result.evals["score"]
                best_lmfn = candidate_fn

            return candidate_result

        candidates: list[CandidateMetricResult] = []
        for _ in range(rounds):
            candidates += pmap(
                run_candidate_zenbase,
                self.candidates(best_lmfn, samples),
                concurrency=concurrency,
            )

        return self.Result(best_lmfn, candidates)

    async def aperform(
        self,
        lmfn: LMFunction[Inputs, Outputs],
        evaluator: CandidateMetricEvaluator[Inputs, Outputs],
        samples: int = 0,
        rounds: int = 1,
        concurrency: int = 1,
    ) -> Result:
        return await asyncify(self.perform)(
            lmfn, evaluator, samples, rounds, concurrency
        )

    def candidates(self, _lmfn: LMFunction[Inputs, Outputs], samples: int):
        max_samples = factorial(len(self.demoset))
        if samples > max_samples:
            log.warn(
                "samples >= factorial(len(demoset)), using factorial(len(demoset))",
                max_samples=max_samples,
                samples=samples,
            )
            samples = max_samples

        for _ in range(samples):
            demos = tuple(self.random.sample(self.demoset, k=self.shots))
            yield LMZenbase(demos=demos)
