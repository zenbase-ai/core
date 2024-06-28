from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple

from zenbase.optim.base import LMOptim
from zenbase.optim.metric.labeled_few_shot import LabeledFewShot
from zenbase.optim.metric.types import CandidateMetricEvaluator, CandidateMetricResult
from zenbase.types import Inputs, LMDemo, LMFunction, LMZenbase, Outputs
from zenbase.utils import get_logger, ot_tracer

log = get_logger(__name__)


@dataclass(kw_only=True)
class BootstrapFewShot(LMOptim[Inputs, Outputs]):
    class Result(NamedTuple):
        best_function: LMFunction[Inputs, Outputs]
        candidate_results: list[CandidateMetricResult]

    demoset: list[LMDemo[Inputs, Outputs]]
    shots: int = field(default=5)

    def __post_init__(self):
        assert 1 <= self.shots <= len(self.demoset)

    @ot_tracer.start_as_current_span("perform")
    def perform(
        self,
        lmfn: LMFunction[Inputs, Outputs],
        deps: list[LMFunction],
        evaluator: CandidateMetricEvaluator[Inputs, Outputs],
        samples: int = 0,
        rounds: int = 1,
        concurrency: int = 1,
        max_sample_to_include: int = 100,
        zenbase=None,
    ) -> Result:
        assert zenbase is not None, "Zenbase is required for this operation"

        # make the base LabelFewShot teacher lm
        teacher_lm, candidates, best_candidate_result = LabeledFewShot(demoset=self.demoset, shots=self.shots).perform(
            lmfn,
            evaluator=evaluator,
            samples=samples,
            rounds=rounds,
        )

        # make an student lm
        student_lm = teacher_lm.clean_and_duplciate()
        print(student_lm)

        # make the validated demo set
        validated_demo_set = []
        for individual_eval in best_candidate_result.individual_evals:
            if individual_eval.passed:
                validated_demo_set.append(individual_eval.demo)

        # empty the traces
        zenbase.all_traces = {}

        # run each of the demoset to fill up the traces
        for validated_demo in validated_demo_set:
            teacher_lm(validated_demo.inputs)

        # consolidate the traces to optimized args
        all_traces = zenbase.all_traces

        each_function_inputs = {}

        for trace, trace_value in all_traces.items():
            for function, function_trace in trace_value.items():
                for inside_functions, inside_functions_traces in function_trace.items():
                    print(inside_functions, inside_functions_traces)
                    if inside_functions not in each_function_inputs:
                        each_function_inputs[inside_functions] = [
                            LMDemo(
                                inputs={"question": inside_functions_traces["args"]["request"].inputs},
                                outputs={"answer": inside_functions_traces["output"]},
                            )
                        ]
                    else:
                        each_function_inputs[inside_functions].append(
                            LMDemo(
                                inputs={"question": inside_functions_traces["args"]["request"].inputs},
                                outputs={"answer": inside_functions_traces["output"]},
                            )
                        )

        # make the new teacher lm
        optimized_args = {}
        for function, demos in each_function_inputs.items():
            optimized_args[function] = {
                "args": {
                    "zenbase": LMZenbase(
                        task_demos=demos,
                    )
                }
            }

        def optimized_fn(request):
            with zenbase.trace_context("optimized", f"optimized_layer_0_{datetime.now().isoformat()}", optimized_args):
                return teacher_lm(request)

        optimized_fn(validated_demo_set[0])

        return self.Result(best_function=optimized_fn, candidate_results=candidates)
