from collections import deque
from copy import copy
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Awaitable, Callable
import inspect

from pyee import AsyncIOEventEmitter


from zenbase.utils import asyncify, id_generator, syncify


@dataclass(frozen=True)
class LMDemo[Inputs: dict, Outputs: dict]:
    inputs: Inputs
    outputs: Outputs

    def __hash__(self):
        return hash((frozenset(self.inputs.items()), frozenset(self.outputs.items())))


@dataclass(frozen=True)
class LMZenbase[Inputs: dict, Outputs: dict]:
    # TODO: These are the thing that you can optimize later (for later inspiration)
    # instructions: list[str] = field(default_factory=list)
    # dos: list[str] = field(default_factory=list)
    # donts: list[str] = field(default_factory=list)
    demos: list[LMDemo[Inputs, Outputs]] = field(default_factory=list)


@dataclass(frozen=True)
class LMRequest[Inputs: dict, Outputs: dict]:
    zenbase: LMZenbase[Inputs, Outputs]
    inputs: Inputs = field(default_factory=dict)
    id: str = field(default_factory=id_generator("request"))


@dataclass(frozen=True)
class LMCall[Inputs: dict, Outputs: dict]:
    function: "LMFunction[Inputs, Outputs]"
    request: LMRequest[Inputs, Outputs]
    outputs: Outputs
    id: str = field(default_factory=id_generator("call"))


type SyncDef[Inputs: dict, Outputs: dict] = Callable[
    [LMRequest[Inputs, Outputs]],
    Outputs,
]
type AsyncDef[Inputs: dict, Outputs: dict] = Callable[
    [LMRequest[Inputs, Outputs]],
    Awaitable[Outputs],
]


class LMFunction[Inputs: dict, Outputs: dict]:
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
        self.id = self.gen_id()
        self.fn = fn

        self.__name__ = getattr(fn, "__name__", "zenbase_lm_fn")
        self.__qualname__ = getattr(fn, "__qualname__", "zenbase_lm_fn")
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
        self, request: LMRequest[Inputs, Outputs], response: Outputs
    ) -> Outputs:
        self.history.append(
            LMCall(
                function=self,
                request=request,
                outputs=response,
            ),
        )
        return response

    def __call__(self, inputs: Inputs = {}, *args, **kwargs) -> Outputs:
        request = self.prepare_request(inputs)
        response = syncify(self.fn)(request, *args, **kwargs)
        return self.process_response(request, response)

    async def coroutine(
        self,
        inputs: Inputs = {},
        *args,
        **kwargs,
    ) -> Outputs:
        request = self.prepare_request(inputs)
        response = await asyncify(self.fn)(request, *args, **kwargs)
        return self.process_response(request, response)


def deflm[
    Inputs: dict,
    Outputs: dict,
](
    function: SyncDef[Inputs, Outputs] | AsyncDef[Inputs, Outputs] | None = None,
    zenbase: LMRequest[Inputs, Outputs] | None = None,
) -> LMFunction[Inputs, Outputs]:
    if function is None:
        return partial(deflm, zenbase=zenbase)

    if isinstance(function, LMFunction):
        return function.refine(zenbase)

    return LMFunction(function, zenbase)
