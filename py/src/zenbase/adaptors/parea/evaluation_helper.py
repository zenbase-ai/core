import json
import logging
from dataclasses import asdict
from functools import partial
from inspect import (
    _empty,  # noqa
    signature,
)
from json import JSONDecodeError
from typing import Callable

from langsmith.evaluation._runner import ExperimentResults  # noqa
from parea import Parea
from parea.schemas import ExperimentStatsSchema, ListExperimentUUIDsFilters
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
)

from zenbase.adaptors.base.evaluation_helper import BaseEvaluationHelper
from zenbase.optim.metric.types import (
    CandidateEvalResult,
    CandidateEvaluator,
    IndividualEvalValue,
    OverallEvalValue,
)
from zenbase.types import LMDemo, LMFunction
from zenbase.utils import expand_nested_json, get_logger, random_name_generator

log = get_logger(__name__)


class PareaEvaluationHelper(BaseEvaluationHelper):
    MetricEvaluator = Callable[[dict[str, float]], OverallEvalValue]

    def __init__(self, client=None):
        super().__init__(client)
        self.evaluator_args = None
        self.evaluator_kwargs = None

    def get_evaluator(self, data: str):
        pass

    @staticmethod
    def default_candidate_evals(stats: ExperimentStatsSchema) -> OverallEvalValue:
        return {**stats.avg_scores, "score": sum(stats.avg_scores.values())}

    def _metric_evaluator_generator(
        self,
        *args,
        p: Parea | None = None,
        candidate_evals: MetricEvaluator = default_candidate_evals,
        **kwargs,
    ) -> CandidateEvaluator:
        p = p or Parea()
        assert isinstance(p, Parea)

        base_metadata = kwargs.pop("metadata", {})
        gen_random_name = random_name_generator(kwargs.pop("name", None))

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(max=8),
            before_sleep=before_sleep_log(log, logging.WARN),
        )
        def evaluate_candidate(function: LMFunction) -> CandidateEvalResult:
            # Check if the function has a default value for the 'request' parameter
            # TODO: Needs clean up, this is not the way to do it.
            if (
                "optimized_args_in_fn" in signature(function).parameters
                and "request" in signature(function).parameters
                and signature(function).parameters["request"].default is _empty
            ):
                # Create a new function with 'None' as the default value for 'request'
                function_with_default = partial(function, request=None)
            else:
                # If 'request' already has a default value, use the function as is
                function_with_default = function

            experiment = p.experiment(
                func=function_with_default,
                *args,
                **kwargs,
                name=gen_random_name(),
                metadata={
                    **base_metadata,
                },
            )

            experiment.run()

            if not experiment.experiment_stats:
                raise RuntimeError("Failed to run experiment on Parea")

            experiments = p.list_experiments(
                ListExperimentUUIDsFilters(experiment_name_filter=experiment.experiment_name)
            )
            experiment__uuid = experiments[0].uuid
            print(f"Num. experiments: {len(experiments)}")
            individual_evals = self._experiment_results_to_individual_evals(
                experiment.experiment_stats, experiment__uuid=experiment__uuid
            )
            return CandidateEvalResult(
                function,
                evals=candidate_evals(experiment.experiment_stats),
                individual_evals=individual_evals,
            )

        return evaluate_candidate

    def _experiment_results_to_individual_evals(
        self,
        experiment_stats: ExperimentStatsSchema,
        threshold=0.5,
        score_name="",
        experiment__uuid=None,
    ) -> list[IndividualEvalValue]:
        if experiment_stats is None or experiment__uuid is None:
            raise ValueError("experiment_stats and experiment__uuid must not be None")

        individual_evals = []
        # Retrieve the JSON logs for the experiment using its UUID
        try:
            json_traces = self._get_experiment_logs(experiment__uuid)
        except JSONDecodeError:
            raise ValueError("Failed to parse experiment logs")

        def find_input_output_with_trace_id(trace_id):
            for trace in json_traces:
                try:
                    if trace["trace_id"] == trace_id:
                        inputs = expand_nested_json(trace["inputs"])
                        outputs = expand_nested_json(trace["output"])
                        for k, v in inputs.items():
                            if isinstance(v, dict) and "zenbase" in v:
                                return v["inputs"], outputs
                except KeyError:
                    continue
            return None, None

        for res in experiment_stats.parent_trace_stats:
            # Skip this iteration if there are no scores associated with the current result
            if not res.scores:
                continue

            # Find the score, prioritizing scores that match the given score name, or defaulting to the first score
            score = next((i.score for i in res.scores if score_name and i.name == score_name), res.scores[0].score)

            if not res.trace_id or not json_traces:
                raise ValueError("Trace ID or logs not found in experiment results")

            inputs, outputs = find_input_output_with_trace_id(res.trace_id)

            if not inputs or not outputs:
                continue

            individual_evals.append(
                IndividualEvalValue(
                    passed=score >= threshold,
                    response=outputs,
                    demo=LMDemo(inputs=inputs, outputs=outputs, adaptor_object=res),
                    score=score,
                )
            )
        return individual_evals

    def _get_experiment_logs(self, experiment__uuid):
        from parea.client import GET_EXPERIMENT_LOGS_ENDPOINT

        filter_data = {"filter_field": None, "filter_operator": None, "filter_value": None}
        endpoint = GET_EXPERIMENT_LOGS_ENDPOINT.format(experiment_uuid=experiment__uuid)
        response = self.client._client.request("POST", endpoint, data=filter_data)  # noqa
        return response.json()

    @classmethod
    def metric_evaluator(
        cls,
        *args,
        p: Parea | None = None,
        candidate_evals: MetricEvaluator = default_candidate_evals,
        **kwargs,
    ) -> CandidateEvaluator:
        # TODO: should
        p = p or Parea()
        assert isinstance(p, Parea)

        base_metadata = kwargs.pop("metadata", {})
        gen_random_name = random_name_generator(kwargs.pop("name", None))

        def evaluate_candidate(function: LMFunction) -> CandidateEvalResult:
            experiment = p.experiment(
                func=function,
                *args,
                **kwargs,
                name=gen_random_name(),
                metadata={
                    **base_metadata,
                    "zenbase": json.dumps(asdict(function.zenbase)),
                },
            )

            experiment.run()
            assert experiment.experiment_stats, "failed to run experiment"

            return CandidateEvalResult(
                function,
                evals=candidate_evals(experiment.experiment_stats),
            )

        return evaluate_candidate
