from typing import Awaitable, Callable, NotRequired, TypedDict


type LMFunction[I, O] = Callable[[I], Awaitable[O]]


class LMFunctionDemo[I, O](TypedDict):
    inputs: I
    outputs: O


class LMFunctionRun[I, O](LMFunctionDemo[I, O]):
    metadata: NotRequired[dict]
    evals: NotRequired[dict]


class LMPrompt[I, O](TypedDict):
    system: NotRequired[str]
    instructions: NotRequired[str]
    examples: list[LMFunctionDemo[I, O]]


type LMEvaluator[I, O] = Callable[
    [LMFunction[I, O], LMPrompt[I, O]],
    Awaitable[LMEvaluatorRun[I, O]],
]

type LMScorer[I, O] = Callable[[list[LMFunctionRun[I, O]]], float]


class LMEvaluatorRun[I, O](TypedDict):
    prompt: LMPrompt[I, O]
    evals: dict
    metadata: NotRequired[dict]
    runs: list[LMFunctionRun[I, O]]


class LMOptimizerRun[I, O](TypedDict):
    winner: LMEvaluatorRun[I, O]
    candidates: list[LMEvaluatorRun[I, O]]
