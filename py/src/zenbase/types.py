from collections import deque
from copy import copy
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Awaitable, Callable, Generic, TypeVar
import inspect


from zenbase.utils import asyncify, id_generator, syncify


Inputs = TypeVar("Inputs", covariant=True, bound=dict)
Outputs = TypeVar("Outputs", covariant=True, bound=dict)


@dataclass(frozen=True)
class LMDemo(Generic[Inputs, Outputs]):
    inputs: Inputs
    outputs: Outputs

    def __hash__(self):
        return hash((frozenset(self.inputs.items()), frozenset(self.outputs.items())))


@dataclass(frozen=True)
class LMZenbase(Generic[Inputs, Outputs]):
    task_demos: list[LMDemo[Inputs, Outputs]] = field(default_factory=list)
    model_params: dict = field(default_factory=dict)  # OpenAI-compatible model params


@dataclass(frozen=True)
class LMRequest(Generic[Inputs, Outputs]):
    zenbase: LMZenbase[Inputs, Outputs]
    inputs: Inputs = field(default_factory=dict)
    id: str = field(default_factory=id_generator("request"))


@dataclass(frozen=True)
class LMResponse(Generic[Outputs]):
    outputs: Outputs
    attributes: dict = field(
        default_factory=dict
    )  # token_count, cost, inference_time, etc.
    id: str = field(default_factory=id_generator("response"))


@dataclass(frozen=True)
class LMCall(Generic[Inputs, Outputs]):
    function: "LMFunction[Inputs, Outputs]"
    request: LMRequest[Inputs, Outputs]
    response: LMResponse[Outputs]
    id: str = field(default_factory=id_generator("call"))


SyncDef = Callable[
    [LMRequest[Inputs, Outputs]],
    Outputs,
]

AsyncDef = Callable[
    [LMRequest[Inputs, Outputs]],
    Awaitable[Outputs],
]


class LMFunction(Generic[Inputs, Outputs]):
    gen_id = staticmethod(id_generator("fn"))

    id: str
    fn: SyncDef[Inputs, Outputs] | AsyncDef[Inputs, Outputs]
    __name__: str
    __qualname__: str
    __doc__: str
    __signature__: inspect.Signature
    zenbase: LMZenbase[Inputs, Outputs]
    history: deque[LMCall[Inputs, Outputs]]

    def __init__(
        self,
        fn: SyncDef[Inputs, Outputs] | AsyncDef[Inputs, Outputs],
        zenbase: LMZenbase | None = None,
        maxhistory: int = 1,
    ):
        self.fn = fn

        if qualname := getattr(fn, "__qualname__", None):
            self.id = qualname
            self.__qualname__ = qualname
        else:
            self.id = self.gen_id()
            self.__qualname__ = f"zenbase_{self.id}"

        self.__name__ = getattr(fn, "__name__", f"zenbase_{self.id}")

        self.__doc__ = getattr(fn, "__doc__", "")
        self.__signature__ = inspect.signature(fn)

        self.zenbase = zenbase or LMZenbase()
        self.history = deque([], maxlen=maxhistory)

    def refine(self, zenbase: LMZenbase | None = None) -> "LMFunction[Inputs, Outputs]":
        dup = copy(self)
        dup.id = self.gen_id()
        dup.zenbase = zenbase or replace(self.zenbase)
        dup.history = deque([], maxlen=self.history.maxlen)
        return dup

    def prepare_request(self, inputs: Inputs) -> LMRequest[Inputs, Outputs]:
        return LMRequest(zenbase=self.zenbase, inputs=inputs)

    def process_response(
        self,
        request: LMRequest[Inputs, Outputs],
        outputs: Outputs,
    ) -> Outputs:
        self.history.append(LMCall(self, request, LMResponse(outputs)))
        return outputs

    def __call__(self, inputs: Inputs, *args, **kwargs) -> Outputs:
        request = self.prepare_request(inputs)
        response = syncify(self.fn)(request, *args, **kwargs)
        return self.process_response(request, response)

    async def coroutine(
        self,
        inputs: Inputs,
        *args,
        **kwargs,
    ) -> Outputs:
        request = self.prepare_request(inputs)
        response = await asyncify(self.fn)(request, *args, **kwargs)
        return self.process_response(request, response)


def deflm(
    function: SyncDef[Inputs, Outputs] | AsyncDef[Inputs, Outputs] | None = None,
    zenbase: LMZenbase[Inputs, Outputs] | None = None,
) -> LMFunction[Inputs, Outputs]:
    if function is None:
        return partial(deflm, zenbase=zenbase)

    if isinstance(function, LMFunction):
        return function.refine(zenbase)

    return LMFunction(function, zenbase)
