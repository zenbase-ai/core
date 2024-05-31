from typing import Callable
import asyncio

from phoenix.evals import LLMEvaluator, run_evals
from zenbase.types import (
    LMEvaluator,
    LMEvaluatorRun,
    LMFunctionRun,
    LMPrompt,
    LMFunction,
)
import pandas as pd


class ArizeZen:
    @classmethod
    def evaluator[
        I, O
    ](
        cls,
        golden_dataframe: pd.DataFrame,
        evaluators: list[LLMEvaluator],
        scorer: Callable[[pd.DataFrame], float],
        provide_explanation: bool = False,
        use_function_calling_if_available: bool = False,
        verbose: bool = False,
        concurrency: int = 20,
    ) -> LMEvaluator[I, O]:
        async def evaluate(
            fn: LMFunction[I, O], prompt: LMPrompt[I, O]
        ) -> LMEvaluatorRun[I, O]:
            candidate_dataframe = golden_dataframe.copy()
            concurrency_lock = asyncio.Semaphore(concurrency)

            async def run_fn(input: I) -> O:
                async with concurrency_lock:
                    return await fn(input)

            candidate_dataframe["attributes.output.value"] = await asyncio.gather(
                *[run_fn(input) for input in golden_dataframe["attributes.input.value"]]
            )

            eval_result_dfs = run_evals(
                candidate_dataframe,
                evaluators,
                provide_explanation,
                use_function_calling_if_available,
                verbose,
                concurrency,
            )

            return LMEvaluatorRun(
                prompt=prompt,
                evals={  # aggregate score for the entire goldens dataset
                    "score": scorer(eval_result_dfs)  # to maximize (DSPy "metric")
                    # can add other key/values here
                },
                metadata={},  # optional dict
                runs=[
                    LMFunctionRun(
                        inputs=r["attributes.input.value"],
                        outputs=r["attributes.output.value"],
                        metadata=r["attributes.metadata.value"],
                        evals={
                            # TODO: look up the evals from eval_result_dfs?
                        },
                    )
                    for r in candidate_dataframe.to_dict(orient="records")
                ],  # list[LMFunctionRun[I, O]]
            )

        return evaluate
