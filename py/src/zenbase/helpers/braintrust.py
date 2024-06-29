from asyncio import Task
from dataclasses import asdict
from typing import AsyncIterator, Awaitable, Callable, Iterator

from braintrust import (
    Eval,
    EvalCase,
    EvalHooks,
    EvalScorer,
    Input,
    Metadata,
    Output,
    ReporterDef,
)

from zenbase.optim.metric.types import CandidateEvalResult
from zenbase.types import LMFunction
from zenbase.utils import random_name_generator


class ZenBraintrust:
    @staticmethod
    def metric_evaluator(
        name: str,
        data: Callable[[], Iterator[EvalCase] | AsyncIterator[EvalCase]],
        task: Callable[[Input, EvalHooks], Output | Awaitable[Output]],
        scores: list[EvalScorer],
        experiment_name: str | None = None,
        trial_count: int = 1,
        metadata: Metadata | None = None,
        is_public: bool = False,
        update: bool = False,
        reporter: ReporterDef | str | None = None,
    ):
        gen_random_name = random_name_generator(experiment_name)

        def evaluate_candidate(function: LMFunction) -> CandidateEvalResult:
            eval_result = Eval(
                name=name,
                experiment_name=gen_random_name(),
                data=data,
                task=task,
                scores=scores,
                trial_count=trial_count,
                metadata={
                    **metadata,
                    **asdict(function.zenbase),
                },
                is_public=is_public,
                update=update,
                reporter=reporter,
            )

            if isinstance(eval_result, Task):
                eval_result = eval_result.result()

            assert eval_result is not None, "Failed to run Braintrust Eval"

            evals = {s.name: s.score for s in eval_result.summary.scores.values()}
            if "score" not in evals:
                evals["score"] = sum(evals.values())

            return CandidateEvalResult(function, evals)

        return evaluate_candidate
