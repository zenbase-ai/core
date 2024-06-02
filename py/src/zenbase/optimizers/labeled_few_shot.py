from functools import wraps
from os import getenv
from random import Random
from typing import Generator

from ..types import (
    LMEvaluator,
    LMEvaluatorRun,
    LMFunctionDemo,
    LMOptimizerRun,
    LMFunction,
    LMPrompt,
)


class LabeledFewShot:
    @staticmethod
    def candidates(
        demos: list[LMFunctionDemo],
        shots: int = 5,
        samples: int = 100,
        seed: int | None = None,
    ) -> Generator[LMPrompt, None, None]:
        assert len(demos) >= shots, "Not enough examples to train the predictor"

        if seed is None:
            seed = int(getenv("RANDOM_SEED", 42))

        rng = Random(seed)
        seen = set()

        for _ in range(samples):
            examples = rng.sample(demos, k=shots)
            if examples in seen:
                seen.add(examples)
                yield LMPrompt(examples=examples)

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
