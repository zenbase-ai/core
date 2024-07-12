from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import Any, Dict, NamedTuple

import cloudpickle

from zenbase.adaptors.arize import ZenArizeAdaptor
from zenbase.adaptors.langchain import ZenLangSmith
from zenbase.core.managers import ZenbaseTracer
from zenbase.optim.base import LMOptim
from zenbase.optim.metric.labeled_few_shot import LabeledFewShot
from zenbase.optim.metric.types import CandidateEvalResult
from zenbase.types import Inputs, LMDemo, LMFunction, LMZenbase, Outputs
from zenbase.utils import get_logger, ot_tracer

log = get_logger(__name__)


@dataclass(kw_only=True)
class BootstrapFewShot(LMOptim[Inputs, Outputs]):
    class Result(NamedTuple):
        best_function: LMFunction[Inputs, Outputs]
        candidate_results: list[CandidateEvalResult] | None = None

    shots: int = field(default=5)
    training_set_demos: list[LMDemo[Inputs, Outputs]] | None = None
    training_set: Any = None  # TODO: it needs to be more generic and pass our Dataset Object here
    test_set: Any = None
    validation_set: Any = None
    base_evaluation = None
    best_evaluation = None
    optimizer_args: Dict[str, dict[str, dict[str, LMDemo]]] = field(default_factory=dict)
    zen_adaptor: Any = None
    evaluator_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.training_set_demos = self.zen_adaptor.fetch_dataset_demos(self.training_set)
        self.zen_adaptor.set_evaluator_kwargs(**self.evaluator_kwargs)
        assert 1 <= self.shots <= len(self.training_set_demos)

    @ot_tracer.start_as_current_span("perform")
    def perform(
        self,
        student_lm: LMFunction[Inputs, Outputs],
        teacher_lm: LMFunction[Inputs, Outputs] | None = None,
        samples: int = 5,
        rounds: int = 1,
        trace_manager: ZenbaseTracer = None,
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

        test_set_evaluator = self.zen_adaptor.get_evaluator(data=self.test_set)
        self.base_evaluation = test_set_evaluator(student_lm)

        if not teacher_lm:
            # Create the base LabeledFewShot teacher model
            teacher_lm = self._create_teacher_model(self.zen_adaptor, student_lm, samples, rounds)

        # Evaluate and validate the demo set
        validated_training_set_demos = self._validate_demo_set(self.zen_adaptor, teacher_lm)

        # Run each validated demo to fill up the traces
        trace_manager.all_traces = {}
        self._run_validated_demos(teacher_lm, validated_training_set_demos)

        # Consolidate the traces to optimized args
        optimized_args = self._consolidate_traces_to_optimized_args(trace_manager)
        self.set_optimizer_args(optimized_args)

        # Create the optimized function
        optimized_fn = self._create_optimized_function(student_lm, optimized_args, trace_manager)

        # Evaluate the optimized function
        self.best_evaluation = test_set_evaluator(optimized_fn)

        return self.Result(best_function=optimized_fn)

    def _create_teacher_model(
        self, zen_adaptor: ZenLangSmith, student_lm: LMFunction, samples: int, rounds: int
    ) -> LMFunction:
        evaluator = zen_adaptor.get_evaluator(data=self.validation_set)
        teacher_lm, _, _ = LabeledFewShot(demoset=self.training_set_demos, shots=self.shots).perform(
            student_lm, evaluator=evaluator, samples=samples, rounds=rounds
        )
        return teacher_lm

    def _validate_demo_set(self, zen_adaptor: ZenLangSmith, teacher_lm: LMFunction) -> list[LMDemo]:
        # TODO: here is an issue that we are not removing the actual training set from the task demo
        #  so it is possible of over fitting but it is not a big issue for now,
        #  we should remove them in the trace_manager
        # def teacher_lm_tweaked(request: LMRequest):
        #     #check inputs in the task demos
        #     for demo in request.zenbase.task_demos:
        #         if request.inputs == demo.inputs:
        #             request.zenbase.task_demos.pop(demo)
        #     return teacher_lm(request)

        # get evaluator for the training set
        evaluate_demo_set = zen_adaptor.get_evaluator(data=self.training_set)
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

    def _consolidate_traces_to_optimized_args(
        self, trace_manager: ZenbaseTracer
    ) -> dict[str, dict[str, dict[str, LMDemo]]]:
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
                    if isinstance(input_args, dict):
                        input_args = {k: str(v).replace("{", " ").replace("}", " ") for k, v in input_args.items()}
                    if isinstance(output_args, dict):
                        output_args = {k: str(v).replace("{", " ").replace("}", " ") for k, v in output_args.items()}

                    if isinstance(self.zen_adaptor, ZenArizeAdaptor):
                        # TODO: Not the right place to do it, clean it up later, but arize needs to get inputs
                        #  and outputs everywhere
                        input_args = {"inputs": input_args}
                        output_args = {"outputs": output_args}

                    each_function_inputs.setdefault(inside_functions, []).append(
                        LMDemo(inputs=input_args, outputs=output_args)
                    )

        optimized_args = {
            function: {"args": {"zenbase": LMZenbase(task_demos=demos)}}
            for function, demos in each_function_inputs.items()
        }
        return optimized_args

    @staticmethod
    def _create_optimized_function(
        student_lm: LMFunction, optimized_args: dict, trace_manager: ZenbaseTracer
    ) -> LMFunction:
        """
        Create the optimized function that will be used to optimize the student function

        :param student_lm: The student function that needs to be optimized
        :param optimized_args: The optimized args that will be used to optimize the student function
        :param trace_manager: The trace manager that will be used to trace the function
        """

        def optimized_fn_base(request, zenbase, optimized_args_in_fn, trace_manager, *args, **kwargs):
            if request is None and "inputs" not in kwargs.keys():
                raise ValueError("Request or inputs should be passed")
            elif request is None:
                request = kwargs["inputs"]
                kwargs.pop("inputs")

            new_optimized_args = deepcopy(optimized_args_in_fn)
            with trace_manager.trace_context(
                "optimized", f"optimized_layer_0_{datetime.now().isoformat()}", new_optimized_args
            ):
                if request is None:
                    return student_lm(*args, **kwargs)
                return student_lm(request, *args, **kwargs)

        optimized_fn = partial(
            optimized_fn_base,
            zenbase=LMZenbase(),  # it doesn't do anything, it is just for type safety
            optimized_args_in_fn=optimized_args,
            trace_manager=trace_manager,
        )
        return optimized_fn

    def set_optimizer_args(self, args: Dict[str, Any]) -> None:
        """
        Set the optimizer arguments.

        :param args: A dictionary containing the optimizer arguments
        """
        self.optimizer_args = args

    def get_optimizer_args(self) -> Dict[str, Any]:
        """
        Get the current optimizer arguments.

        :return: A dictionary containing the current optimizer arguments
        """
        return self.optimizer_args

    def save_optimizer_args(self, file_path: str) -> None:
        """
        Save the optimizer arguments to a dill file.

        :param file_path: The path to save the dill file
        """
        with open(file_path, "wb") as f:
            cloudpickle.dump(self.optimizer_args, f)

    @classmethod
    def load_optimizer_and_function(
        cls, optimizer_args_file: str, student_lm: LMFunction[Inputs, Outputs], trace_manager: ZenbaseTracer
    ) -> LMFunction[Inputs, Outputs]:
        """
        Load optimizer arguments and create an optimized function.

        :param optimizer_args_file: The path to the JSON file containing optimizer arguments
        :param student_lm: The student function to be optimized
        :param trace_manager: The trace manager to be used
        :return: An optimized function
        """
        optimizer_args = cls._load_optimizer_args(optimizer_args_file)
        return cls._create_optimized_function(student_lm, optimizer_args, trace_manager)

    @classmethod
    def _load_optimizer_args(cls, file_path: str) -> Dict[str, Any]:
        """
        Load optimizer arguments from a dill file.

        :param file_path: The path to load the dill file from
        :return: A dictionary containing the loaded optimizer arguments
        """
        with open(file_path, "rb") as f:
            return cloudpickle.load(f)
