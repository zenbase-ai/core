from typing import TYPE_CHECKING, Callable

import pandas as pd

from zenbase.optim.metric.types import CandidateEvalResult, OverallEvalValue
from zenbase.types import LMDemo, LMFunction
from zenbase.utils import amap

if TYPE_CHECKING:
    from phoenix.evals import LLMEvaluator


class ZenPhoenix:
    MetricEvaluator = Callable[[list["LLMEvaluator"], list[pd.DataFrame]], OverallEvalValue]

    @staticmethod
    def df_to_demos(df: pd.DataFrame) -> list[LMDemo]:
        raise NotImplementedError()

    @staticmethod
    def default_metric(evaluators: list["LLMEvaluator"], eval_dfs: list[pd.DataFrame]) -> OverallEvalValue:
        evals = {"score": sum(df.score.mean() for df in eval_dfs)}
        evals.update({e.__name__: df.score.mean() for e, df in zip(evaluators, eval_dfs)})
        return evals

    @classmethod
    def metric_evaluator(
        cls,
        dataset: pd.DataFrame,
        evaluators: list["LLMEvaluator"],
        metric_evals: MetricEvaluator = default_metric,
        concurrency: int = 20,
        *args,
        **kwargs,
    ):
        from phoenix.evals import run_evals

        async def run_experiment(function: LMFunction) -> CandidateEvalResult:
            nonlocal dataset
            run_df = dataset.copy()
            # TODO: Is it typical for there to only be 1 value?
            responses = await amap(
                function,
                run_df["attributes.input.value"].to_list(),  # i don't think this works
                concurrency=concurrency,
            )
            run_df["attributes.output.value"] = responses

            eval_dfs = run_evals(run_df, evaluators, *args, concurrency=concurrency, **kwargs)

            return CandidateEvalResult(
                function,
                evals=metric_evals(evaluators, eval_dfs),
            )

        return run_experiment
