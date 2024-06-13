from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from random import Random

from pyee.asyncio import AsyncIOEventEmitter

from zenbase.types import LMFunction
from zenbase.utils import random_factory


@dataclass(kw_only=True)
class LMOptim[Inputs: dict, Outputs: dict](ABC):
    random: Random = field(default_factory=random_factory)
    events: AsyncIOEventEmitter = field(default_factory=AsyncIOEventEmitter)

    @abstractmethod
    def perform(
        self,
        lmfn: LMFunction[Inputs, Outputs],
        *args,
        **kwargs,
    ): ...

    @abstractmethod
    async def aperform(
        self,
        lmfn: LMFunction[Inputs, Outputs],
        *args,
        **kwargs,
    ) -> LMFunction[Inputs, Outputs]: ...
