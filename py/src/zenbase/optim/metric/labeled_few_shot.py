from dataclasses import dataclass, field
from math import factorial
from typing import NamedTuple

from zenbase.optim.base import LMOptim
from zenbase.optim.metric.types import CandidateEvalResult, CandidateEvaluator
from zenbase.types import Inputs, LMDemo, LMFunction, LMZenbase, Outputs
from zenbase.utils import asyncify, get_logger, ksuid, ot_tracer, pmap, posthog

log = get_logger(__name__)


@dataclass(kw_only=True)
class LabeledFewShot(LMOptim[Inputs, Outputs]):
    class Result(NamedTuple):
        best_function: LMFunction[Inputs, Outputs]
        candidate_results: list[CandidateEvalResult]
        best_candidate_result: CandidateEvalResult | None

    demoset: list[LMDemo[Inputs, Outputs]]
    shots: int = field(default=5)

    def __post_init__(self):
        assert 1 <= self.shots <= len(self.demoset)

    @ot_tracer.start_as_current_span("perform")
    def perform(
        self,
        lmfn: LMFunction[Inputs, Outputs],
        evaluator: CandidateEvaluator[Inputs, Outputs],
        samples: int = 0,
        rounds: int = 1,
        concurrency: int = 1,
    ) -> Result:
        samples = samples or len(self.demoset)

        best_score = float("-inf")
        best_lmfn = lmfn
        best_candidate_result = None

        @ot_tracer.start_as_current_span("run_experiment")
        def run_candidate_zenbase(zenbase: LMZenbase):
            nonlocal best_score, best_lmfn, best_candidate_result

            candidate_fn = lmfn.clean_and_duplicate(zenbase)
            try:
                candidate_result = evaluator(candidate_fn)
            except Exception as e:
                log.error("candidate evaluation failed", error=e)
                candidate_result = CandidateEvalResult(candidate_fn, {"score": float("-inf")})

            self.events.emit("candidate", candidate_result)

            if candidate_result.evals["score"] > best_score:
                best_score = candidate_result.evals["score"]
                best_lmfn = candidate_fn
                best_candidate_result = candidate_result

            return candidate_result

        candidates: list[CandidateEvalResult] = []
        for _ in range(rounds):
            candidates += pmap(
                run_candidate_zenbase,
                self.candidates(best_lmfn, samples),
                concurrency=concurrency,
            )

        posthog().capture(
            distinct_id=ksuid(),
            event="optimize_labeled_few_shot",
            properties={
                "evals": {c.function.id: c.evals for c in candidates},
            },
        )

        return self.Result(best_lmfn, candidates, best_candidate_result)

    async def aperform(
        self,
        lmfn: LMFunction[Inputs, Outputs],
        evaluator: CandidateEvaluator[Inputs, Outputs],
        samples: int = 0,
        rounds: int = 1,
        concurrency: int = 1,
    ) -> Result:
        return await asyncify(self.perform)(lmfn, evaluator, samples, rounds, concurrency)

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
            yield LMZenbase(task_demos=demos)
