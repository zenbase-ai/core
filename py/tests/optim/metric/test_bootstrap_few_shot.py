from unittest.mock import Mock, mock_open, patch

import pytest

from zenbase.adaptors.langchain import ZenLangSmith
from zenbase.core.managers import ZenbaseTracer
from zenbase.optim.metric.bootstrap_few_shot import BootstrapFewShot
from zenbase.types import LMDemo, LMFunction, LMZenbase


@pytest.fixture
def mock_zen_adaptor():
    adaptor = Mock(spec=ZenLangSmith)
    adaptor.fetch_dataset_demos.return_value = [
        LMDemo(inputs={"input": "test1"}, outputs={"output": "result1"}),
        LMDemo(inputs={"input": "test2"}, outputs={"output": "result2"}),
        LMDemo(inputs={"input": "test3"}, outputs={"output": "result3"}),
    ]
    adaptor.get_evaluator.return_value = Mock(
        return_value=Mock(
            individual_evals=[
                Mock(passed=True, demo=LMDemo(inputs={"input": "test1"}, outputs={"output": "result1"})),
                Mock(passed=True, demo=LMDemo(inputs={"input": "test2"}, outputs={"output": "result2"})),
                Mock(passed=False, demo=LMDemo(inputs={"input": "test3"}, outputs={"output": "result3"})),
            ]
        )
    )
    return adaptor


@pytest.fixture
def mock_trace_manager():
    return Mock(spec=ZenbaseTracer)


@pytest.fixture
def bootstrap_few_shot(mock_zen_adaptor):
    return BootstrapFewShot(
        shots=2,
        training_set=Mock(),
        test_set=Mock(),
        validation_set=Mock(),
        zen_adaptor=mock_zen_adaptor,
    )


def test_init(bootstrap_few_shot):
    assert bootstrap_few_shot.shots == 2
    assert len(bootstrap_few_shot.training_set_demos) == 3


def test_init_invalid_shots():
    with pytest.raises(AssertionError):
        BootstrapFewShot(shots=0, training_set=Mock(), test_set=Mock(), validation_set=Mock(), zen_adaptor=Mock())


def test_create_teacher_model(bootstrap_few_shot, mock_zen_adaptor):
    mock_lmfn = Mock(spec=LMFunction)
    with patch("zenbase.optim.metric.bootstrap_few_shot.LabeledFewShot") as mock_labeled_few_shot:
        mock_labeled_few_shot.return_value.perform.return_value = (Mock(), None, None)
        teacher_model = bootstrap_few_shot._create_teacher_model(mock_zen_adaptor, mock_lmfn, 5, 1)
        assert teacher_model is not None
        mock_labeled_few_shot.assert_called_once()


def test_validate_demo_set(bootstrap_few_shot, mock_zen_adaptor):
    mock_teacher_lm = Mock(spec=LMFunction)
    validated_demos = bootstrap_few_shot._validate_demo_set(mock_zen_adaptor, mock_teacher_lm)
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


def test_consolidate_traces_to_optimized_args(mock_trace_manager, bootstrap_few_shot):
    mock_trace_manager.all_traces = {
        "trace1": {
            "func1": {
                "inner_func1": {"args": {"request": Mock(inputs={"input": "test1"})}, "output": {"output": "result1"}}
            }
        }
    }
    optimized_args = bootstrap_few_shot._consolidate_traces_to_optimized_args(mock_trace_manager)
    assert "inner_func1" in optimized_args
    assert isinstance(optimized_args["inner_func1"]["args"]["zenbase"], LMZenbase)


def test_create_optimized_function():
    mock_student_lm = Mock(spec=LMFunction)
    mock_trace_manager = Mock(spec=ZenbaseTracer)
    optimized_args = {"func1": {"args": {"zenbase": LMZenbase(task_demos=[])}}}

    optimized_fn = BootstrapFewShot._create_optimized_function(mock_student_lm, optimized_args, mock_trace_manager)
    assert callable(optimized_fn)


@patch("zenbase.optim.metric.bootstrap_few_shot.partial")
def test_perform(mock_partial, bootstrap_few_shot, mock_zen_adaptor, mock_trace_manager):
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
                        )

    assert isinstance(result, BootstrapFewShot.Result)
    assert result.best_function is not None


def test_set_and_get_optimizer_args(bootstrap_few_shot):
    test_args = {"test": "args"}
    bootstrap_few_shot.set_optimizer_args(test_args)
    assert bootstrap_few_shot.get_optimizer_args() == test_args


@patch("cloudpickle.dump")
def test_save_optimizer_args(mock_dump, bootstrap_few_shot, tmp_path):
    test_args = {"test": "args"}
    bootstrap_few_shot.set_optimizer_args(test_args)
    file_path = tmp_path / "test_optimizer_args.dill"
    bootstrap_few_shot.save_optimizer_args(str(file_path))
    mock_dump.assert_called_once()


@patch("builtins.open", new_callable=mock_open, read_data="dummy data")
@patch("cloudpickle.load")
def test_load_optimizer_args(mock_load, mock_file):
    test_args = {"test": "args"}
    mock_load.return_value = test_args
    loaded_args = BootstrapFewShot._load_optimizer_args("dummy_path")
    mock_file.assert_called_once_with("dummy_path", "rb")
    mock_load.assert_called_once()
    assert loaded_args == test_args


@patch("zenbase.optim.metric.bootstrap_few_shot.BootstrapFewShot._load_optimizer_args")
@patch("zenbase.optim.metric.bootstrap_few_shot.BootstrapFewShot._create_optimized_function")
def test_load_optimizer_and_function(mock_create_optimized_function, mock_load_optimizer_args):
    mock_student_lm = Mock(spec=LMFunction)
    mock_trace_manager = Mock(spec=ZenbaseTracer)
    mock_load_optimizer_args.return_value = {"test": "args"}
    mock_create_optimized_function.return_value = Mock(spec=LMFunction)

    result = BootstrapFewShot.load_optimizer_and_function("dummy_path", mock_student_lm, mock_trace_manager)

    mock_load_optimizer_args.assert_called_once_with("dummy_path")
    mock_create_optimized_function.assert_called_once()
    assert isinstance(result, Mock)
    assert isinstance(result, LMFunction)
