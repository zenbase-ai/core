import instructor
import pytest
from instructor import Instructor
from openai import OpenAI
from pydantic import BaseModel

from zenbase.core.managers import ZenbaseTracer
from zenbase.predefined.generic_lm_function.optimizer import GenericLMFunctionOptimizer


class InputModel(BaseModel):
    question: str


class OutputModel(BaseModel):
    answer: str


@pytest.fixture(scope="module")
def openai_client() -> OpenAI:
    return OpenAI()


@pytest.fixture(scope="module")
def instructor_client(openai_client: OpenAI) -> Instructor:
    return instructor.from_openai(openai_client)


@pytest.fixture(scope="module")
def zenbase_tracer() -> ZenbaseTracer:
    return ZenbaseTracer()


@pytest.fixture
def generic_optimizer(instructor_client, zenbase_tracer):
    training_set = [
        {"inputs": {"question": "What is the capital of France?"}, "outputs": {"answer": "Paris"}},
        {"inputs": {"question": "Who wrote Romeo and Juliet?"}, "outputs": {"answer": "William Shakespeare"}},
        {"inputs": {"question": "What is the largest planet in our solar system?"}, "outputs": {"answer": "Jupiter"}},
        {"inputs": {"question": "Who painted the Mona Lisa?"}, "outputs": {"answer": "Leonardo da Vinci"}},
        {"inputs": {"question": "What is the chemical symbol for gold?"}, "outputs": {"answer": "Au"}},
    ]

    return GenericLMFunctionOptimizer(
        instructor_client=instructor_client,
        prompt="You are a helpful assistant. Answer the user's question concisely.",
        input_model=InputModel,
        output_model=OutputModel,
        model="gpt-4o-mini",
        zenbase_tracer=zenbase_tracer,
        training_set=training_set,
        validation_set=[
            {"inputs": {"question": "What is the capital of Italy?"}, "outputs": {"answer": "Rome"}},
            {"inputs": {"question": "What is the capital of France?"}, "outputs": {"answer": "Paris"}},
            {"inputs": {"question": "What is the capital of Germany?"}, "outputs": {"answer": "Berlin"}},
        ],
        test_set=[
            {"inputs": {"question": "Who invented the telephone?"}, "outputs": {"answer": "Alexander Graham Bell"}},
            {"inputs": {"question": "Who is CEO of microsoft?"}, "outputs": {"answer": "Bill Gates"}},
            {"inputs": {"question": "Who is founder of Facebook?"}, "outputs": {"answer": "Mark Zuckerberg"}},
        ],
        shots=len(training_set),  # Set shots to the number of training examples
    )


@pytest.mark.helpers
def test_generic_optimizer_optimize(generic_optimizer):
    result = generic_optimizer.optimize()
    assert result is not None
    assert isinstance(result, GenericLMFunctionOptimizer.Result)
    assert result.best_function is not None
    assert callable(result.best_function)
    assert isinstance(result.candidate_results, list)
    assert result.best_candidate_result is not None

    # Check base evaluation
    assert generic_optimizer.base_evaluation is not None

    # Check best evaluation
    assert generic_optimizer.best_evaluation is not None

    # Test the best function
    test_input = InputModel(question="What is the capital of Italy?")
    output = result.best_function(test_input)
    assert isinstance(output, OutputModel)
    assert isinstance(output.answer, str)
    assert output.answer.strip().lower() == "rome"


@pytest.mark.helpers
def test_generic_optimizer_evaluations(generic_optimizer):
    result = generic_optimizer.optimize()

    # Check that base and best evaluations exist
    assert generic_optimizer.base_evaluation is not None
    assert generic_optimizer.best_evaluation is not None

    # Additional checks to ensure the structure of the result
    assert isinstance(result, GenericLMFunctionOptimizer.Result)
    assert result.best_function is not None
    assert isinstance(result.candidate_results, list)
    assert result.best_candidate_result is not None


@pytest.mark.helpers
def test_generic_optimizer_custom_evaluator(instructor_client, zenbase_tracer):
    def custom_evaluator(output: OutputModel, ideal_output: dict) -> dict:
        return {"passed": int(output.answer.lower() == ideal_output["answer"].lower()), "length": len(output.answer)}

    training_set = [
        {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}},
        {"inputs": {"question": "What is the capital of France?"}, "outputs": {"answer": "Paris"}},
        {"inputs": {"question": "Who wrote Romeo and Juliet?"}, "outputs": {"answer": "William Shakespeare"}},
        {"inputs": {"question": "What is the largest planet in our solar system?"}, "outputs": {"answer": "Jupiter"}},
        {"inputs": {"question": "Who painted the Mona Lisa?"}, "outputs": {"answer": "Leonardo da Vinci"}},
    ]

    optimizer = GenericLMFunctionOptimizer(
        instructor_client=instructor_client,
        prompt="You are a helpful assistant. Answer the user's question concisely.",
        input_model=InputModel,
        output_model=OutputModel,
        model="gpt-4o-mini",
        zenbase_tracer=zenbase_tracer,
        training_set=training_set,
        validation_set=[{"inputs": {"question": "What is 3+3?"}, "outputs": {"answer": "6"}}],
        test_set=[{"inputs": {"question": "What is 4+4?"}, "outputs": {"answer": "8"}}],
        custom_evaluator=custom_evaluator,
        shots=len(training_set),  # Set shots to the number of training examples
    )

    result = optimizer.optimize()
    assert result is not None
    assert isinstance(result, GenericLMFunctionOptimizer.Result)
    assert "length" in optimizer.best_evaluation.individual_evals[0].details

    # Test the custom evaluator
    test_input = InputModel(question="What is 5+5?")
    output = result.best_function(test_input)
    assert isinstance(output, OutputModel)
    assert isinstance(output.answer, str)

    # Manually apply the custom evaluator
    eval_result = custom_evaluator(output, {"answer": "10"})
    assert "passed" in eval_result
    assert "length" in eval_result


@pytest.mark.helpers
def test_create_lm_function_with_demos(generic_optimizer):
    prompt = "You are a helpful assistant. Answer the user's question concisely."
    demos = [
        {"inputs": {"question": "What is the capital of France?"}, "outputs": {"answer": "Paris"}},
        {"inputs": {"question": "Who wrote Romeo and Juliet?"}, "outputs": {"answer": "William Shakespeare"}},
    ]

    lm_function = generic_optimizer.create_lm_function_with_demos(prompt, demos)

    # Test that the function is created and can be called
    test_input = InputModel(question="What is the capital of Italy?")
    result = lm_function(test_input)

    assert isinstance(result, OutputModel)
    assert isinstance(result.answer, str)
    assert result.answer.strip().lower() == "rome"

    # Test with a question from the demos
    test_input_demo = InputModel(question="What is the capital of France?")
    result_demo = lm_function(test_input_demo)

    assert isinstance(result_demo, OutputModel)
    assert isinstance(result_demo.answer, str)
    assert result_demo.answer.strip().lower() == "paris"
