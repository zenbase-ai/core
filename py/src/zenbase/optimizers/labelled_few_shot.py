from functools import wraps
from itertools import permutations
from os import getenv
from random import Random
from typing import Awaitable

from ..types import Evaluator, EvaluatorRun, FunctionDemo, OptimizerRun, Predictor


async def labelled_few_shot_optimizer[
    I, O
](
    predictor: Predictor[I, O],
    evaluator: Evaluator[I, O],
    examples: list[FunctionDemo[I, O]],
    shots: int = 5,
    samples: int = 100,
    seed: int | None = None,
) -> tuple[Predictor[I, O], OptimizerRun]:
    if seed is None:
        seed = int(getenv("RANDOM_SEED", 42))

    example_permutations = list(permutations(examples, shots))
    shuffled_examples = Random(seed).shuffle(example_permutations)

    eval_runs: list[EvaluatorRun] = []
    for _, examples in zip(range(samples), shuffled_examples):
        eval_runs.append(await evaluator(predictor, {"examples": examples}))

    best_run = max(eval_runs, key=lambda x: x["eval"]["score"])

    @wraps(predictor)
    def best_predictor(inputs: dict) -> Awaitable[dict]:
        return predictor({**inputs, **best_run["candidate"]})

    return best_predictor, OptimizerRun(winner=best_run, candidates=eval_runs)
