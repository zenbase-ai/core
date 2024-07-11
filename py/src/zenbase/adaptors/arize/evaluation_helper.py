from zenbase.adaptors.base.evaluation_helper import BaseEvaluationHelper
from zenbase.optim.metric.types import CandidateEvalResult, CandidateEvaluator, IndividualEvalValue
from zenbase.types import LMDemo, LMFunction
from zenbase.utils import random_name_generator


class ArizeEvaluationHelper(BaseEvaluationHelper):
    def get_evaluator(self, data: str):
        evaluator_kwargs_to_pass = self.evaluator_kwargs.copy()
        dataset = self.client.get_dataset(name=data)
        evaluator_kwargs_to_pass.update({"dataset": dataset})
        return self._metric_evaluator_generator(**evaluator_kwargs_to_pass)

    def _metric_evaluator_generator(self, threshold: float = 0.5, **evaluate_kwargs) -> CandidateEvaluator:
        from phoenix.experiments import run_experiment

        gen_random_name = random_name_generator(evaluate_kwargs.pop("experiment_prefix", None))

        def evaluate_candidate(function: LMFunction) -> CandidateEvalResult:
            def arize_adapted_function(input):
                return function(input)

            experiment = run_experiment(
                evaluate_kwargs["dataset"],
                arize_adapted_function,
                experiment_name=gen_random_name(),
                evaluators=evaluate_kwargs.get("evaluators", None),
            )
            list_of_individual_evals = []
            for individual_eval in experiment.eval_runs:
                example_id = experiment.runs[individual_eval.experiment_run_id].dataset_example_id
                example = experiment.dataset.examples[example_id]
                if individual_eval.result:
                    list_of_individual_evals.append(
                        IndividualEvalValue(
                            passed=individual_eval.result.score >= threshold,
                            response=experiment.runs[individual_eval.experiment_run_id].output,
                            demo=LMDemo(
                                inputs=example.input,
                                outputs=example.output,
                            ),
                            score=individual_eval.result.score,
                        )
                    )

            # make average scores of all evaluation metrics
            avg_scores = [i.stats["avg_score"][0] for i in experiment.eval_summaries]
            avg_score = sum(avg_scores) / len(avg_scores)

            return CandidateEvalResult(function, {"score": avg_score}, individual_evals=list_of_individual_evals)

        return evaluate_candidate

    @classmethod
    def metric_evaluator(cls, threshold: float = 0.5, **evaluate_kwargs) -> CandidateEvaluator:
        # TODO: Should remove and deprecate
        from phoenix.experiments import run_experiment

        gen_random_name = random_name_generator(evaluate_kwargs.pop("experiment_prefix", None))

        def evaluate_candidate(function: LMFunction) -> CandidateEvalResult:
            def arize_adapted_function(input):
                return function(input)

            experiment = run_experiment(
                evaluate_kwargs["dataset"],
                arize_adapted_function,
                experiment_name=gen_random_name(),
                evaluators=evaluate_kwargs.get("evaluators", None),
            )
            list_of_individual_evals = []
            for individual_eval in experiment.eval_runs:
                example_id = experiment.runs[individual_eval.experiment_run_id].dataset_example_id
                example = experiment.dataset.examples[example_id]

                list_of_individual_evals.append(
                    IndividualEvalValue(
                        passed=individual_eval.result.score >= threshold,
                        response=experiment.runs[individual_eval.experiment_run_id].output,
                        demo=LMDemo(
                            inputs=example.input,
                            outputs=example.output,
                        ),
                        score=individual_eval.result.score,
                    )
                )

            # make average scores of all evaluation metrics
            avg_scores = [i.stats["avg_score"][0] for i in experiment.eval_summaries]
            avg_score = sum(avg_scores) / len(avg_scores)

            return CandidateEvalResult(function, {"score": avg_score}, individual_evals=list_of_individual_evals)

        return evaluate_candidate
