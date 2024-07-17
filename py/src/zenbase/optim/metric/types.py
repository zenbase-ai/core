import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Generic, TypedDict

from zenbase.types import Dataclass, Inputs, LMDemo, LMFunction, Outputs


class OverallEvalValue(TypedDict):
    score: float


@dataclasses.dataclass(frozen=True)
class IndividualEvalValue(Dataclass, Generic[Outputs]):
    passed: bool
    response: Outputs
    demo: LMDemo
    score: float | None = None
    details: dict = field(default_factory=dict)


@dataclass
class CandidateEvalResult(Generic[Inputs, Outputs]):
    function: LMFunction[Inputs, Outputs]
    evals: OverallEvalValue = field(default_factory=dict)
    individual_evals: list[IndividualEvalValue] = field(default_factory=list)


CandidateEvaluator = Callable[
    [LMFunction[Inputs, Outputs]],
    CandidateEvalResult[Inputs, Outputs],
]
