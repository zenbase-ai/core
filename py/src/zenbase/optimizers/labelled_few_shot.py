from functools import wraps
from itertools import permutations
from os import getenv
from random import Random
from typing import Generator

from ..types import (
    LMEvaluator,
    LMEvaluatorRun,
    LMFunctionDemo,
    LMOptimizerRun,
    LMFunction,
)


class LabelledFewShot:
    @staticmethod
    def candidates[
        I, O
    ](
        demos: list[LMFunctionDemo[I, O]],
        shots: int = 5,
        samples: int = 100,
        seed: int | None = None,
    ) -> Generator[LMFunctionDemo[I, O], None, None]:
        assert len(demos) >= shots, "Not enough examples to train the predictor"

        if seed is None:
            seed = int(getenv("RANDOM_SEED", 42))

        example_sets = list(permutations(demos, shots))
        Random(seed).shuffle(example_sets)

        for _, demos in zip(range(samples), example_sets):
            yield {"examples": list(demos)}

    @classmethod
    async def optimize[
        I, O
    ](
        cls,
        fn: LMFunction[I, O],
        evaluator: LMEvaluator[I, O],
        demos: list[LMFunctionDemo[I, O]],
        shots: int = 5,
        samples: int = 100,
        seed: int | None = None,
    ) -> tuple[LMFunction[I, O], LMOptimizerRun]:
        eval_runs: list[LMEvaluatorRun] = []

        for prompt in cls.candidates(demos, shots, samples, seed):
            eval_runs.append(await evaluator(fn, prompt))

        best_run = max(eval_runs, key=lambda r: r["evals"]["score"])

        @wraps(fn)
        async def best_fn(**kwargs):
            return await fn(**kwargs, prompt=best_run["prompt"])

        return best_fn, LMOptimizerRun(winner=best_run, candidates=eval_runs)
