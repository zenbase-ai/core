from dataclasses import asdict
from typing import TYPE_CHECKING

from langsmith.evaluation._runner import ExperimentResults  # noqa

from zenbase.adaptors.base.evaluation_helper import BaseEvaluationHelper
from zenbase.optim.metric.types import CandidateEvalResult, CandidateEvaluator, IndividualEvalValue, OverallEvalValue
from zenbase.types import LMDemo, LMFunction
from zenbase.utils import random_name_generator

if TYPE_CHECKING:
    from langsmith import schemas


class LangsmithEvaluationHelper(BaseEvaluationHelper):
    def __init__(self, client=None):
        super().__init__(client)
        self.evaluator_args = None
        self.evaluator_kwargs = None

    def set_evaluator_kwargs(self, *args, **kwargs) -> None:
        self.evaluator_kwargs = kwargs
        self.evaluator_args = args

    def get_evaluator(self, data: "schemas.Dataset") -> CandidateEvaluator:
        evaluator_kwargs_to_pass = self.evaluator_kwargs.copy()
        evaluator_kwargs_to_pass.update({"data": data.name})
        return self._metric_evaluator_generator(**evaluator_kwargs_to_pass)

    def _metric_evaluator_generator(self, threshold: float = 0.5, **evaluate_kwargs) -> CandidateEvaluator:
        from langsmith import evaluate

        metadata = evaluate_kwargs.pop("metadata", {})
        gen_random_name = random_name_generator(evaluate_kwargs.pop("experiment_prefix", None))

        def evaluate_candidate(function: LMFunction) -> CandidateEvalResult:
            experiment_results = evaluate(
                function,
                experiment_prefix=gen_random_name(),
                metadata={
                    **metadata,
                },
                **evaluate_kwargs,
            )

            individual_evals = self._experiment_results_to_individual_evals(experiment_results, threshold)

            if summary_results := experiment_results._summary_results["results"]:  # noqa
                evals = self._eval_results_to_evals(summary_results)
            else:
                evals = self._individual_evals_to_overall_evals(individual_evals)

            return CandidateEvalResult(function, evals, individual_evals)

        return evaluate_candidate

    @classmethod
    def metric_evaluator(cls, threshold: float = 0.5, **evaluate_kwargs) -> CandidateEvaluator:
        # TODO: Should remove and deprecate
        from langsmith import evaluate

        metadata = evaluate_kwargs.pop("metadata", {})
        gen_random_name = random_name_generator(evaluate_kwargs.pop("experiment_prefix", None))

        def evaluate_candidate(function: LMFunction) -> CandidateEvalResult:
            experiment_results = evaluate(
                function,
                experiment_prefix=gen_random_name(),
                metadata={
                    **metadata,
                    **asdict(function.zenbase),
                },
                **evaluate_kwargs,
            )

            individual_evals = cls._experiment_results_to_individual_evals(experiment_results, threshold)

            if summary_results := experiment_results._summary_results["results"]:  # noqa
                evals = cls._eval_results_to_evals(summary_results)
            else:
                evals = cls._individual_evals_to_overall_evals(individual_evals)

            return CandidateEvalResult(function, evals, individual_evals)

        return evaluate_candidate

    @staticmethod
    def _individual_evals_to_overall_evals(individual_evals: list[IndividualEvalValue]) -> OverallEvalValue:
        if not individual_evals:
            raise ValueError("No evaluation results")

        if individual_evals[0].score is not None:
            number_of_scores = sum(1 for e in individual_evals if e.score is not None)
            score = sum(e.score for e in individual_evals if e.score is not None) / number_of_scores
        else:
            number_of_filled_passed = sum(1 for e in individual_evals if e.passed is not None)
            number_of_actual_passed = sum(1 for e in individual_evals if e.passed)
            score = number_of_actual_passed / number_of_filled_passed

        return {"score": score}

    @staticmethod
    def _experiment_results_to_individual_evals(
        experiment_results: ExperimentResults, threshold=0.5
    ) -> (list)[IndividualEvalValue]:
        individual_evals = []
        for res in experiment_results._results:  # noqa
            if not res["evaluation_results"]["results"]:
                continue
            score = res["evaluation_results"]["results"][0].score
            inputs = res["example"].inputs
            outputs = res["example"].outputs
            individual_evals.append(
                IndividualEvalValue(
                    passed=score >= threshold,
                    response=outputs,
                    demo=LMDemo(inputs=inputs, outputs=outputs, adaptor_object=res["example"]),
                    details=res,
                    score=score,
                )
            )
        return individual_evals

    @staticmethod
    def _eval_results_to_evals(eval_results: list) -> OverallEvalValue:
        if not eval_results:
            raise ValueError("No evaluation results")

        return {
            "score": eval_results[0].score,
            **{r.key: r.dict() for r in eval_results},
        }
