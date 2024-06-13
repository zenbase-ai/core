from typing import Callable, TYPE_CHECKING

import pandas as pd

from zenbase.optim.metric.types import MetricEvals, CandidateMetricResult
from zenbase.types import LMDemo, LMFunction
from zenbase.utils import amap


if TYPE_CHECKING:
    from phoenix.evals import LLMEvaluator


class ZenPhoenix:
    type MetricEvaluator = Callable[
        [list["LLMEvaluator"], list[pd.DataFrame]], MetricEvals
    ]

    @staticmethod
    def df_to_demos[
        Inputs: dict, Outputs: dict
    ](df: pd.DataFrame) -> list[LMDemo[Inputs, Outputs]]:
        raise NotImplementedError()

    @staticmethod
    def default_metric(
        evaluators: list["LLMEvaluator"], eval_dfs: list[pd.DataFrame]
    ) -> MetricEvals:
        evals = {"score": sum(df.score.mean() for df in eval_dfs)}
        evals.update(
            {e.__name__: df.score.mean() for e, df in zip(evaluators, eval_dfs)}
        )
        return evals

    @classmethod
    def metric_evaluator[
        Inputs: dict,
        Outputs: dict,
    ](
        cls,
        dataset: pd.DataFrame,
        evaluators: list["LLMEvaluator"],
        metric_evals: MetricEvaluator = default_metric,
        concurrency: int = 20,
        *args,
        **kwargs,
    ):
        from phoenix.evals import run_evals

        async def run_experiment(
            function: LMFunction[Inputs, Outputs],
        ) -> CandidateMetricResult[Inputs, Outputs]:
            nonlocal dataset
            run_df = dataset.copy()
            # TODO: Is it typical for there to only be 1 value?
            responses = await amap(
                function,
                run_df["attributes.input.value"].to_list(),  # i don't think this works
                concurrency=concurrency,
            )
            run_df["attributes.output.value"] = responses

            eval_dfs = run_evals(
                run_df, evaluators, *args, concurrency=concurrency, **kwargs
            )

            return CandidateMetricResult(
                function,
                evals=metric_evals(evaluators, eval_dfs),
            )

        return run_experiment
