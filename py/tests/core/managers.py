from collections import OrderedDict
from datetime import datetime
from unittest.mock import patch

import pytest

from zenbase.core.managers import ZenbaseTracer
from zenbase.types import LMFunction, LMZenbase


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


@pytest.fixture
def tracer():
    return ZenbaseTracer(max_traces=3)


def test_init(tracer):
    assert isinstance(tracer.all_traces, OrderedDict)
    assert tracer.max_traces == 3
    assert tracer.current_trace is None
    assert tracer.current_key is None
    assert tracer.optimized_args == {}


def test_flush(tracer):
    tracer.all_traces = OrderedDict({"key1": "value1", "key2": "value2"})
    tracer.flush()
    assert len(tracer.all_traces) == 0


def test_add_trace(tracer):
    # Add first trace
    tracer.add_trace("timestamp1", "func1", {"data": "trace1"})
    assert len(tracer.all_traces) == 1
    assert "timestamp1" in tracer.all_traces

    # Add second trace
    tracer.add_trace("timestamp2", "func2", {"data": "trace2"})
    assert len(tracer.all_traces) == 2

    # Add third trace
    tracer.add_trace("timestamp3", "func3", {"data": "trace3"})
    assert len(tracer.all_traces) == 3

    # Add fourth trace (should remove oldest)
    tracer.add_trace("timestamp4", "func4", {"data": "trace4"})
    assert len(tracer.all_traces) == 3
    assert "timestamp1" not in tracer.all_traces
    assert "timestamp4" in tracer.all_traces


@patch("zenbase.utils.ksuid")
def test_trace_function(mock_ksuid, tracer):
    mock_ksuid.return_value = "test_timestamp"

    def test_func(request):
        return request.inputs[0] + request.inputs[1]

    zenbase = LMZenbase()
    traced_func = tracer.trace_function(test_func, zenbase)
    assert isinstance(traced_func, LMFunction)

    result = traced_func(inputs=(2, 3))

    assert result == 5
    trace = tracer.all_traces[list(tracer.all_traces.keys())[0]]
    assert "test_func" in trace["test_func"]
    trace_info = trace["test_func"]["test_func"]
    assert trace_info["args"]["request"].inputs == (2, 3)
    assert trace_info["output"] == 5


def test_trace_context(tracer):
    with tracer.trace_context("test_func", "test_timestamp"):
        assert tracer.current_key == "test_timestamp"
        assert isinstance(tracer.current_trace, dict)

    assert tracer.current_trace is None
    assert tracer.current_key is None
    assert "test_timestamp" in tracer.all_traces


def test_max_traces_limit(tracer):
    for i in range(5):
        tracer.add_trace(f"timestamp{i}", f"func{i}", {"data": f"trace{i}"})

    assert len(tracer.all_traces) == 3
    assert "timestamp0" not in tracer.all_traces
    assert "timestamp1" not in tracer.all_traces
    assert "timestamp2" in tracer.all_traces
    assert "timestamp3" in tracer.all_traces
    assert "timestamp4" in tracer.all_traces


@patch("zenbase.utils.ksuid")
def test_optimized_args(mock_ksuid, tracer):
    mock_ksuid.return_value = "test_timestamp"

    def test_func(request, z=3):
        x, y = request.inputs
        return x + y + z

    tracer.optimized_args = {"test_func": {"args": {"z": 5}}}
    zenbase = LMZenbase()
    traced_func = tracer.trace_function(test_func, zenbase)

    result = traced_func(inputs=(2, 10))

    assert result == 17  # 2 + 10 + 5
    trace = tracer.all_traces[list(tracer.all_traces.keys())[0]]
    assert "test_func" in trace["test_func"]
    trace_info = trace["test_func"]["test_func"]
    assert trace_info["args"]["request"].inputs == (2, 10)
    assert trace_info["args"]["z"] == 5
    assert trace_info["output"] == 17
