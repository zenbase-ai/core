from dataclasses import dataclass, field
from typing import Callable, TypedDict

from zenbase.types import LMFunction


class MetricEvals(TypedDict):
    score: float


@dataclass
class CandidateMetricResult[Inputs: dict, Outputs: dict]:
    function: LMFunction[Inputs, Outputs]
    evals: MetricEvals = field(default_factory=dict)


type CandidateMetricEvaluator[Inputs: dict, Outputs: dict] = Callable[
    [LMFunction[Inputs, Outputs]],
    CandidateMetricResult[Inputs, Outputs],
]
