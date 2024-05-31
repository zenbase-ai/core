from typing import Sequence, Any
import asyncio
import datetime
import uuid

from asyncer import asyncify
from langsmith import Client, evaluate, schemas
from langsmith.evaluation._runner import (
    _get_random_name,
    EVALUATOR_T,
    SUMMARY_EVALUATOR_T,
)
from sorcery import dict_of

from zenbase.types import (
    LMPrompt,
    LMEvaluator,
    LMEvaluatorRun,
    LMFunctionDemo,
    LMFunctionRun,
    LMFunction,
)


class LangSmithZen:
    @staticmethod
    def examples(
        dataset_id: str | uuid.UUID | None = None,
        dataset_name: str | None = None,
        example_ids: Sequence[str | uuid.UUID] | None = None,
        as_of: datetime.datetime | str | None = None,
        splits: Sequence[str] | None = None,
        inline_s3_urls: bool = True,
        limit: int | None = None,
        metadata: dict | None = None,
        client: Client | None = None,
        **kwargs: Any,
    ) -> list[LMFunctionDemo]:
        client = client or Client()
        return [
            LMFunctionDemo(inputs=e.inputs, outputs=e.outputs)
            for e in client.list_examples(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                example_ids=example_ids,
                as_of=as_of,
                splits=splits,
                inline_s3_urls=inline_s3_urls,
                limit=limit,
                metadata=metadata,
                **kwargs,
            )
        ]

    @classmethod
    def evaluator[
        I, O
    ](
        cls,
        data: str | uuid.UUID | Sequence[schemas.Example],
        evaluators: Sequence[EVALUATOR_T],
        summary_evaluators: Sequence[SUMMARY_EVALUATOR_T],
        metadata: dict | None = None,
        description: str | None = None,
        max_concurrency: int | None = None,
        num_repetitions: int = 1,
        client: Client | None = None,
    ) -> LMEvaluator[I, O]:
        assert any(summary_evaluators), "At least one summary evaluator is required"

        base_metadata = metadata or {}
        evaluate_params = dict_of(
            data,
            client,
            evaluators,
            summary_evaluators,
            description,
            max_concurrency,
            num_repetitions,
        )

        async def evaluator(
            fn: LMFunction[I, O],
            prompt: LMPrompt[I, O],
        ) -> LMEvaluatorRun[I, O]:
            zenbase_params = {"prompt": prompt}
            metadata = {**base_metadata, **zenbase_params}

            eval_results = await asyncify(evaluate)(
                lambda inputs: asyncio.run(fn(**inputs, **zenbase_params)),
                metadata=metadata,
                experiment_prefix=f"zenbase-{_get_random_name()}",
                **evaluate_params,
            )
            runs = [
                LMFunctionRun(
                    inputs=res["run"].inputs["inputs"],
                    outputs=res["run"].outputs,
                    metadata={
                        **res["run"].metadata,
                        "golden": res["example"].outputs,
                    },
                    evals=cls._eval_results_to_evals_dict(res["evaluation_results"]),
                )
                for res in eval_results._results
            ]

            return LMEvaluatorRun(
                prompt=prompt,
                runs=runs,
                metadata=metadata,
                evals=cls._eval_results_to_evals_dict(eval_results._summary_results),
            )

        return evaluator

    @classmethod
    def _eval_results_to_evals_dict(cls, eval_results) -> dict:
        return {
            **{r.key: r.dict() for r in eval_results["results"]},
            "score": eval_results["results"][0].score,
        }
