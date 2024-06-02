from typing import Awaitable, Callable, NotRequired, ParamSpecKwargs, TypeVar, TypedDict
from inspect import iscoroutinefunction

from asyncer import asyncify

I = ParamSpecKwargs("I")  # noqa: E741
O = TypeVar("O")  # noqa: E741


class LMFunction:
    def __init__(self, function: Callable[[I.args, *I.kwargs], Awaitable[O]]):
        self._function = (
            function if iscoroutinefunction(function) else asyncify(function)
        )

    async def __call__(self, *args: I.args, **kwargs: I.kwargs) -> O:
        return await self._function(*args, **kwargs)


class LMFunctionDemo(TypedDict):
    inputs: I
    outputs: O


class LMFunctionRun(LMFunctionDemo):
    metadata: NotRequired[dict]
    evals: NotRequired[dict]


class LMPrompt(TypedDict):
    system: NotRequired[str]
    instructions: NotRequired[str]
    examples: list[LMFunctionDemo]


type LMEvaluator = Callable[
    [LMFunction, LMPrompt],
    Awaitable[LMEvaluatorRun],
]

type LMScorer = Callable[[list[LMFunctionRun]], float]


class LMEvaluatorRun(TypedDict):
    prompt: LMPrompt
    evals: dict
    metadata: NotRequired[dict]
    runs: list[LMFunctionRun[I, O]]


class LMOptimizerRun(TypedDict):
    winner: LMEvaluatorRun
    candidates: list[LMEvaluatorRun]
