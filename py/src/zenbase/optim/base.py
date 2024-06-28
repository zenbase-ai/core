from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from random import Random
from typing import Generic

from pyee.asyncio import AsyncIOEventEmitter

from zenbase.types import Inputs, LMFunction, Outputs
from zenbase.utils import random_factory


@dataclass(kw_only=True)
class LMOptim(Generic[Inputs, Outputs], ABC):
    random: Random = field(default_factory=random_factory)
    events: AsyncIOEventEmitter = field(default_factory=AsyncIOEventEmitter)

    @abstractmethod
    def perform(
        self,
        lmfn: LMFunction[Inputs, Outputs],
        *args,
        **kwargs,
    ): ...
