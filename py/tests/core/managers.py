from datetime import datetime

import pytest

from zenbase.core.managers import ZenbaseTracer


@pytest.fixture
def zenbase_manager():
    return ZenbaseTracer()


@pytest.fixture
def layer_2_1(zenbase_manager):
    @zenbase_manager.trace_function
    def _layer_2_1(request, instruction="default_instruction_2_1", candidates=[]):
        layer_2_1_output = f"layer_2_1_output_{str(request.inputs)}"
        return layer_2_1_output

    return _layer_2_1


@pytest.fixture
def layer_1_1(zenbase_manager, layer_2_1):
    @zenbase_manager.trace_function
    def _layer_1_1(request, instruction="default_instruction_1_1", candidates=[]):
        layer_2_1(request.inputs, instruction=instruction)
        layer_1_1_output = f"layer_1_1_output_{str(request.inputs)}"
        return layer_1_1_output

    return _layer_1_1


@pytest.fixture
def layer_1_2(zenbase_manager):
    @zenbase_manager.trace_function
    def _layer_1_2(request, instruction="default_instruction_1_2", candidates=[]):
        layer_1_2_output = f"layer_1_2_output_{str(request.inputs)}"
        return layer_1_2_output

    return _layer_1_2


@pytest.fixture
def layer_0(zenbase_manager, layer_1_1, layer_1_2):
    @zenbase_manager.trace_function
    def _layer_0(request, instruction="default_instruction_0", candidates=[]):
        layer_1_1(inputs=request.inputs)
        layer_1_2(inputs=request.inputs)
        layer_0_output = f"layer_0_output_{str(request.inputs)}"
        return layer_0_output

    return _layer_0


@pytest.fixture
def layer_0_2(zenbase_manager, layer_1_1, layer_1_2):
    @zenbase_manager.trace_function
    def _layer_0_2(request, instruction="default_instruction_0_2", candidates=[]):
        layer_1_1(inputs=request.inputs["inputs"])
        layer_1_2(inputs=request.inputs["inputs"])
        layer_0_output = f"layer_0_2_output_{str(request.inputs['inputs'])}"
        return layer_0_output

    return _layer_0_2


def test_trace_layer_0(zenbase_manager, layer_0):
    inputs = [{"inputs": i} for i in range(5)]

    for inputs in inputs:
        layer_0(inputs=inputs)

    assert len(zenbase_manager.all_traces) == 5
    for trace in zenbase_manager.all_traces.values():
        assert "_layer_0" in trace


def test_trace_layer_0_multiple_runs(zenbase_manager, layer_0):
    inputs = [{"inputs": i} for i in range(5)]

    for the_input in inputs:
        layer_0(inputs=the_input)
    for the_input in inputs:
        layer_0(inputs=the_input)

    assert len(zenbase_manager.all_traces) == 10


def test_trace_layer_0_2(zenbase_manager, layer_0_2):
    inputs = [{"inputs": i} for i in range(5)]

    for the_input in inputs:
        layer_0_2(inputs=the_input)

    assert len(zenbase_manager.all_traces) == 5
    for trace in zenbase_manager.all_traces.values():
        assert "_layer_0_2" in trace


def test_trace_layer_0_with_optimized_args(zenbase_manager, layer_0):
    inputs = [{"inputs": i} for i in range(5)]
    optimized_args = {
        "layer_2_1": {"args": {"instruction": "optimized_instruction_2_1", "candidates": ["optimized_candidate_2_1"]}},
        "layer_1_1": {"args": {"instruction": "optimized_instruction_1_1", "candidates": ["optimized_candidate_1_1"]}},
        "layer_1_2": {"args": {"instruction": "optimized_instruction_1_2", "candidates": ["optimized_candidate_1_2"]}},
        "layer_0": {"args": {"instruction": "optimized_instruction_0", "candidates": ["optimized_candidate_0"]}},
    }

    def optimized_layer_0(*args, **kwargs):
        with zenbase_manager.trace_context(
            "optimized_layer_0", f"optimized_layer_0_{datetime.now().isoformat()}", optimized_args
        ):
            return layer_0(*args, **kwargs)

    for the_input in inputs:
        optimized_layer_0(inputs=the_input)

    assert len(zenbase_manager.all_traces) == 5
    for trace in zenbase_manager.all_traces.values():
        assert "optimized_layer_0" in trace


def test_trace_layer_functions(zenbase_manager, layer_2_1, layer_1_1, layer_1_2):
    inputs = [{"inputs": i} for i in range(5)]

    for inputs in inputs:
        layer_2_1(inputs=inputs)
        layer_1_1(inputs=inputs)
        layer_1_2(inputs=inputs)

    assert len(zenbase_manager.all_traces) == 15
    for trace in zenbase_manager.all_traces.values():
        assert any(func in trace for func in ["_layer_2_1", "_layer_1_1", "_layer_1_2"])
