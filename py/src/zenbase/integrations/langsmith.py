from typing import Sequence

from asyncer import asyncify
from langsmith import Client
from langsmith.evaluation import (
    evaluate,
    EVALUATOR_T,
    SUMMARY_EVALUATOR_T,
    ExperimentResults,
)
from sorcery import dict_of

from zenbase.types import (
    Candidate,
    Evaluator,
    EvaluatorRun,
    FunctionDemo,
    FunctionRun,
    Predictor,
    Scorer,
)


def langsmith_examples(
    train_dataset: str,
    client: Client | None = None,
) -> list[FunctionDemo]:
    client = client or Client()
    return [
        {"inputs": e.inputs, "outputs": e.outputs}
        for e in client.list_examples(dataset_name=train_dataset)
    ]


def langsmith_evaluator[
    I, O
](
    test_dataset: str,
    scorer: Scorer,
    evaluators: Sequence[EVALUATOR_T],
    summary_evaluators: Sequence[SUMMARY_EVALUATOR_T],
    metadata: dict | None = None,
    description: str | None = None,
    max_concurrency: int | None = None,
    num_repetitions: int = 1,
) -> Evaluator[I, O]:
    evaluate_params = dict_of(
        evaluators, summary_evaluators, description, max_concurrency, num_repetitions
    )

    async def evaluator(
        predictor: Predictor[I, O], candidate: Candidate[I, O]
    ) -> EvaluatorRun[I, O]:
        zenbase_params = {"zenbase": candidate}

        eval_results = await asyncify(evaluate)(
            lambda inputs: predictor({**inputs, **zenbase_params}),
            data=test_dataset,
            metadata={**metadata, **zenbase_params},
            **evaluate_params,
        )

        function_runs = _results_to_runs(eval_results)

        return EvaluatorRun(
            candidate=candidate,
            function_runs=function_runs,
            metadata={
                **metadata,
                **zenbase_params,
            },
            eval={
                **eval_results._summary_results,
                "score": scorer(function_runs),
            },
        )

    return evaluator


def _results_to_runs(experiment_results: ExperimentResults) -> list[FunctionRun]:
    return [
        FunctionRun(
            inputs=res["run"].inputs,
            outputs=res["run"].outputs,
            metadata=res["run"].metadata,
            eval=res["evaluation_results"],
        )
        for res in experiment_results
    ]
