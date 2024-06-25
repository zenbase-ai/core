from dataclasses import dataclass, field
from typing import NamedTuple

from zenbase.optim.abc import LMOptim
from zenbase.optim.metric.types import CandidateMetricEvaluator, CandidateMetricResult
from zenbase.types import Inputs, LMDemo, LMFunction, Outputs
from zenbase.utils import get_logger

log = get_logger(__name__)


@dataclass(kw_only=True)
class BootstrapFewShot(LMOptim[Inputs, Outputs]):
    class Result(NamedTuple):
        best_function: LMFunction[Inputs, Outputs]
        candidate_results: list[CandidateMetricResult]

    demoset: list[LMDemo[Inputs, Outputs]]
    shots: int = field(default=5)

    def __post_init__(self):
        assert 1 <= self.shots <= len(self.demoset)

    def perform(
        self,
        lmfn: LMFunction[Inputs, Outputs],
        deps: list[LMFunction],
        evaluator: CandidateMetricEvaluator[Inputs, Outputs],
        samples: int = 0,
        rounds: int = 1,
        concurrency: int = 1,
    ) -> Result:
        pass
