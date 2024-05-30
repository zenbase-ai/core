from typing import Awaitable, Callable, NotRequired, TypedDict


type Predictor[I, O] = Callable[[I], Awaitable[O]]


class FunctionDemo[I, O](TypedDict):
    inputs: I
    outputs: O


class FunctionRun[I, O](FunctionDemo[I, O]):
    metadata: NotRequired[dict]
    eval: NotRequired[dict]


class Candidate[I, O](TypedDict):
    instructions: NotRequired[str]
    examples: list[FunctionDemo[I, O]]


type Evaluator[I, O] = Callable[
    [Predictor[I, O], Candidate[I, O]],
    Awaitable[EvaluatorRun[I, O]],
]

type Scorer = Callable[..., float]


class EvaluatorRun[I, O](TypedDict):
    candidate: Candidate[I, O]
    eval: dict
    metadata: dict
    function_runs: list[FunctionRun]


class OptimizerRun[I, O](TypedDict):
    winner: EvaluatorRun[I, O]
    candidates: list[EvaluatorRun[I, O]]
