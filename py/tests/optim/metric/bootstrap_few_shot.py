from unittest.mock import Mock, patch

import pytest

from zenbase.adaptors.langchain import ZenLangSmith
from zenbase.core.managers import TraceManager
from zenbase.optim.metric.bootstrap_few_shot import BootstrapFewShot
from zenbase.types import LMDemo, LMFunction, LMZenbase


@pytest.fixture
def mock_helper_class():
    helper = Mock(spec=ZenLangSmith)
    helper.fetch_dataset_demos.return_value = [
        LMDemo(inputs={"input": "test1"}, outputs={"output": "result1"}),
        LMDemo(inputs={"input": "test2"}, outputs={"output": "result2"}),
        LMDemo(inputs={"input": "test3"}, outputs={"output": "result3"}),
    ]
    helper.get_evaluator.return_value = Mock(
        return_value=Mock(
            individual_evals=[
                Mock(passed=True, demo=LMDemo(inputs={"input": "test1"}, outputs={"output": "result1"})),
                Mock(passed=True, demo=LMDemo(inputs={"input": "test2"}, outputs={"output": "result2"})),
                Mock(passed=False, demo=LMDemo(inputs={"input": "test3"}, outputs={"output": "result3"})),
            ]
        )
    )
    return helper


@pytest.fixture
def mock_trace_manager():
    return Mock(spec=TraceManager)


@pytest.fixture
def bootstrap_few_shot(mock_helper_class):
    return BootstrapFewShot(
        training_set=mock_helper_class.fetch_dataset_demos(),
        shots=2,
        training_set_original=Mock(),
        test_set_original=Mock(),
    )


def test_init(bootstrap_few_shot):
    assert bootstrap_few_shot.shots == 2
    assert len(bootstrap_few_shot.training_set) == 3


def test_init_invalid_shots():
    with pytest.raises(AssertionError):
        BootstrapFewShot(training_set=[LMDemo(inputs={}, outputs={})], shots=0)


def test_create_teacher_model(bootstrap_few_shot, mock_helper_class):
    mock_lmfn = Mock(spec=LMFunction)
    with patch("zenbase.optim.metric.bootstrap_few_shot.LabeledFewShot") as mock_labeled_few_shot:
        mock_labeled_few_shot.return_value.perform.return_value = (Mock(), None, None)
        teacher_model = bootstrap_few_shot._create_teacher_model(mock_helper_class, mock_lmfn, 5, 1)
        assert teacher_model is not None
        mock_labeled_few_shot.assert_called_once()


def test_validate_demo_set(bootstrap_few_shot, mock_helper_class):
    mock_teacher_lm = Mock(spec=LMFunction)
    validated_demos = bootstrap_few_shot._validate_demo_set(mock_helper_class, mock_teacher_lm)
    assert len(validated_demos) == 2
    assert all(demo.inputs["input"].startswith("test") for demo in validated_demos)


def test_run_validated_demos():
    mock_teacher_lm = Mock(spec=LMFunction)
    validated_demo_set = [
        LMDemo(inputs={"input": "test1"}, outputs={"output": "result1"}),
        LMDemo(inputs={"input": "test2"}, outputs={"output": "result2"}),
    ]
    BootstrapFewShot._run_validated_demos(mock_teacher_lm, validated_demo_set)
    assert mock_teacher_lm.call_count == 2


def test_consolidate_traces_to_optimized_args(mock_trace_manager):
    mock_trace_manager.all_traces = {
        "trace1": {
            "func1": {
                "inner_func1": {"args": {"request": Mock(inputs={"input": "test1"})}, "output": {"output": "result1"}}
            }
        }
    }
    optimized_args = BootstrapFewShot._consolidate_traces_to_optimized_args(mock_trace_manager)
    assert "inner_func1" in optimized_args
    assert isinstance(optimized_args["inner_func1"]["args"]["zenbase"], LMZenbase)


def test_create_optimized_function():
    mock_student_lm = Mock(spec=LMFunction)
    mock_trace_manager = Mock(spec=TraceManager)
    optimized_args = {"func1": {"args": {"zenbase": LMZenbase(task_demos=[])}}}

    optimized_fn = BootstrapFewShot._create_optimized_function(mock_student_lm, optimized_args, mock_trace_manager)
    assert callable(optimized_fn)


@patch("zenbase.optim.metric.bootstrap_few_shot.partial")
def test_perform(mock_partial, bootstrap_few_shot, mock_helper_class, mock_trace_manager):
    mock_student_lm = Mock(spec=LMFunction)
    mock_teacher_lm = Mock(spec=LMFunction)

    with patch.object(bootstrap_few_shot, "_create_teacher_model", return_value=mock_teacher_lm):
        with patch.object(bootstrap_few_shot, "_validate_demo_set"):
            with patch.object(bootstrap_few_shot, "_run_validated_demos"):
                with patch.object(bootstrap_few_shot, "_consolidate_traces_to_optimized_args"):
                    with patch.object(bootstrap_few_shot, "_create_optimized_function"):
                        result = bootstrap_few_shot.perform(
                            mock_student_lm,
                            mock_teacher_lm,
                            samples=5,
                            rounds=1,
                            trace_manager=mock_trace_manager,
                            helper_class=mock_helper_class,
                        )

    assert isinstance(result, BootstrapFewShot.Result)
    assert result.best_function is not None
