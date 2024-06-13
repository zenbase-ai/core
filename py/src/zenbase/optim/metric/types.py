from dataclasses import dataclass, field
from typing import Callable, Generic, TypedDict

from zenbase.types import Inputs, LMFunction, Outputs


class MetricEvals(TypedDict):
    score: float


@dataclass
class CandidateMetricResult(Generic[Inputs, Outputs]):
    function: LMFunction[Inputs, Outputs]
    evals: MetricEvals = field(default_factory=dict)


CandidateMetricEvaluator = Callable[
    [LMFunction[Inputs, Outputs]],
    CandidateMetricResult[Inputs, Outputs],
]
