from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

from zenbase.core.managers import TraceManager
from zenbase.helpers.langchain import ZenLangSmith
from zenbase.optim.base import LMOptim
from zenbase.optim.metric.labeled_few_shot import LabeledFewShot
from zenbase.optim.metric.types import CandidateEvalResult
from zenbase.types import Inputs, LMDemo, LMFunction, LMZenbase, Outputs
from zenbase.utils import get_logger, ot_tracer

if TYPE_CHECKING:
    from langsmith import schemas

log = get_logger(__name__)


@dataclass(kw_only=True)
class BootstrapFewShot(LMOptim[Inputs, Outputs]):
    class Result(NamedTuple):
        best_function: LMFunction[Inputs, Outputs]
        candidate_results: list[CandidateEvalResult] | None = None

    training_set: list[LMDemo[Inputs, Outputs]] | None
    shots: int = field(default=5)
    training_set_original: "schemas.Dataset" = None
    test_set_original: "schemas.Dataset" = None
    base_evaluation = None
    best_evaluation = None

    def __post_init__(self):
        assert 1 <= self.shots <= len(self.training_set)

    @ot_tracer.start_as_current_span("perform")
    def perform(
        self,
        student_lm: LMFunction[Inputs, Outputs],
        teacher_lm: LMFunction[Inputs, Outputs] | None = None,
        samples: int = 5,
        rounds: int = 1,
        trace_manager: TraceManager = None,
        helper_class: ZenLangSmith | None = None,
    ) -> Result:
        """
        This function will perform the bootstrap few shot optimization on the given student_lm function.
        It will return the best function that is optimized based on the given student_lm function.


        :param student_lm: The student function that needs to be optimized
        :param teacher_lm: The teacher function that will be used to optimize the student function
        :param samples: The number of samples to be used for the optimization
        :param rounds: The number of rounds to be used for the optimization in the LabeledFewShot
        :param trace_manager: The trace manager that will be used to trace the function
        :param helper_class: The helper class that will be used to fetch the dataset and evaluator
        """
        assert trace_manager is not None, "Zenbase is required for this operation"

        self.training_set = helper_class.fetch_dataset_demos(self.training_set_original)
        test_set_evaluator = helper_class.get_evaluator(data=self.test_set_original.name)
        self.base_evaluation = test_set_evaluator(student_lm)

        if not teacher_lm:
            # Create the base LabeledFewShot teacher model
            teacher_lm = self._create_teacher_model(helper_class, student_lm, samples, rounds)

        # Evaluate and validate the demo set
        validated_demo_set = self._validate_demo_set(helper_class, teacher_lm)

        # Run each validated demo to fill up the traces
        trace_manager.all_traces = {}
        self._run_validated_demos(teacher_lm, validated_demo_set)

        # Consolidate the traces to optimized args
        optimized_args = self._consolidate_traces_to_optimized_args(trace_manager)

        # Create the optimized function
        optimized_fn = self._create_optimized_function(student_lm, optimized_args, trace_manager)

        # Evaluate the optimized function
        self.best_evaluation = test_set_evaluator(optimized_fn)

        return self.Result(best_function=optimized_fn)

    def _create_teacher_model(self, helper_class, lmfn, samples, rounds):
        evaluator = helper_class.get_evaluator()
        teacher_lm, _, _ = LabeledFewShot(demoset=self.training_set, shots=self.shots).perform(
            lmfn, evaluator=evaluator, samples=samples, rounds=rounds
        )
        return teacher_lm

    def _validate_demo_set(self, helper_class, teacher_lm):
        # TODO: here is an issue that we are not removing the actual training set from the task demo
        #  so it is possible of over fitting but it is not a big issue for now,
        #  we should remove them in the trace_manager

        # get evaluator for the training set
        evaluate_demo_set = helper_class.get_evaluator(data=self.training_set_original.name)
        # run the evaluation and get the result of the evaluation
        result = evaluate_demo_set(teacher_lm)
        # find the validated training set that has been passed
        validated_demo_set = [eval.demo for eval in result.individual_evals if eval.passed]
        return validated_demo_set

    @staticmethod
    def _run_validated_demos(teacher_lm: LMFunction, validated_demo_set: list[LMDemo]) -> None:
        """
        Run each of the validated demos to fill up the traces

        :param teacher_lm: The teacher model to run the demos
        :param validated_demo_set: The validated demos to run
        """
        for validated_demo in validated_demo_set:
            teacher_lm(validated_demo.inputs)

    @staticmethod
    def _consolidate_traces_to_optimized_args(trace_manager: TraceManager) -> dict[str, dict[str, dict[str, LMDemo]]]:
        """
        Consolidate the traces to optimized args that will be used to optimize the student function

        :param trace_manager: The trace manager that contains all the traces
        """
        all_traces = trace_manager.all_traces
        each_function_inputs = {}

        for trace_value in all_traces.values():
            for function_trace in trace_value.values():
                for inside_functions, inside_functions_traces in function_trace.items():
                    input_args = inside_functions_traces["args"]["request"].inputs
                    output_args = inside_functions_traces["output"]

                    # Sanitize input and output arguments by replacing curly braces with spaces.
                    # This prevents conflicts when using these arguments as keys in template rendering within LangChain.
                    input_args = {k: v.replace("{", " ").replace("}", " ") for k, v in input_args.items()}
                    output_args = {k: v.replace("{", " ").replace("}", " ") for k, v in output_args.items()}

                    each_function_inputs.setdefault(inside_functions, []).append(
                        LMDemo(inputs=input_args, outputs=output_args)
                    )

        optimized_args = {
            function: {"args": {"zenbase": LMZenbase(task_demos=demos)}}
            for function, demos in each_function_inputs.items()
        }
        return optimized_args

    @staticmethod
    def _create_optimized_function(student_lm, optimized_args, trace_manager):
        """
        Create the optimized function that will be used to optimize the student function

        :param student_lm: The student function that needs to be optimized
        :param optimized_args: The optimized args that will be used to optimize the student function
        :param trace_manager: The trace manager that will be used to trace the function
        """

        def optimized_fn_base(request, zenbase, optimized_args_in_fn, trace_manager):
            new_optimized_args = deepcopy(optimized_args_in_fn)
            with trace_manager.trace_context(
                "optimized", f"optimized_layer_0_{datetime.now().isoformat()}", new_optimized_args
            ):
                return student_lm(request)

        optimized_fn = partial(
            optimized_fn_base,
            zenbase=LMZenbase(),  # it doesn't do anything, it is just for type safety
            optimized_args_in_fn=optimized_args,
            trace_manager=trace_manager,
        )
        return optimized_fn
