import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Generic, TypedDict

from zenbase.types import Dataclass, Inputs, LMDemo, LMFunction, Outputs


class MetricEvals(TypedDict):
    score: float


@dataclasses.dataclass(frozen=True)
class IndividualEvalMetric(Dataclass, Generic[Outputs]):
    passed: bool
    response: Outputs
    details: dict
    demo: LMDemo


@dataclass
class CandidateMetricResult(Generic[Inputs, Outputs]):
    function: LMFunction[Inputs, Outputs]
    evals: MetricEvals = field(default_factory=dict)
    individual_evals: list[IndividualEvalMetric] = field(default_factory=list)


CandidateMetricEvaluator = Callable[
    [LMFunction[Inputs, Outputs]],
    CandidateMetricResult[Inputs, Outputs],
]
