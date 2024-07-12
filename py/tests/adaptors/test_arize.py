import logging
import os

import pytest
from datasets import DatasetDict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from phoenix.session.client import Client

from zenbase.adaptors.arize import ZenArizeAdaptor
from zenbase.core.managers import ZenbaseTracer
from zenbase.optim.metric.bootstrap_few_shot import BootstrapFewShot
from zenbase.optim.metric.labeled_few_shot import LabeledFewShot
from zenbase.settings import TEST_DIR
from zenbase.types import LMRequest
from zenbase.utils import ksuid

SAMPLES = 2
SHOTS = 3
EVALSET_SIZE = 5
TESTSET_SIZE = 2
TRAINSET_SIZE = 5
VALIDATIONSET_SIZE = 2

log = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def arize_phoenix():
    import phoenix as px

    px.launch_app()
    phoenix_client = px.Client()
    return phoenix_client


@pytest.fixture(scope="module")
def openai():
    return OpenAI()


@pytest.fixture(scope="module")
def zen_arize_adaptor(arize_phoenix):
    return ZenArizeAdaptor(arize_phoenix)


def create_dataset_with_examples(zen_arize_adaptor: ZenArizeAdaptor, prefix: str, item_set: list) -> str:
    dataset_name = ksuid(prefix=prefix)

    inputs = [{"question": example["question"]} for example in item_set]
    expected_outputs = [{"answer": example["answer"]} for example in item_set]
    zen_arize_adaptor.add_examples_to_dataset(dataset_name, inputs, expected_outputs)
    return dataset_name


@pytest.fixture(scope="module")
def train_set(gsm8k_dataset: DatasetDict, zen_arize_adaptor: ZenArizeAdaptor):
    return create_dataset_with_examples(
        zen_arize_adaptor,
        "GSM8K_train_set_parea_dataset",
        list(gsm8k_dataset["train"].select(range(TRAINSET_SIZE))),
    )


@pytest.fixture(scope="module")
def validation_set(gsm8k_dataset: DatasetDict, zen_arize_adaptor: ZenArizeAdaptor):
    return create_dataset_with_examples(
        zen_arize_adaptor,
        "GSM8K_validation_set_parea_dataset",
        list(gsm8k_dataset["train"].select(range(TRAINSET_SIZE + 1, TRAINSET_SIZE + VALIDATIONSET_SIZE + 1))),
    )


@pytest.fixture(scope="module")
def test_set(gsm8k_dataset: DatasetDict, zen_arize_adaptor: ZenArizeAdaptor):
    return create_dataset_with_examples(
        zen_arize_adaptor,
        "GSM8K_test_set_parea_dataset",
        list(gsm8k_dataset["test"].select(range(TESTSET_SIZE))),
    )


@pytest.fixture
def optim(gsm8k_demoset: list):
    return LabeledFewShot(demoset=gsm8k_demoset, shots=SHOTS)


@pytest.mark.helpers
def test_create_and_add_examples(arize_phoenix: Client, zen_arize_adaptor: ZenArizeAdaptor):
    inputs = [{"question": "What is 1+1?"}, {"question": "What is 2+2?"}]
    outputs = [{"answer:": "2"}, {"answer": "4"}]
    dataset_name = str(ksuid("gsm8k-test"))
    zen_arize_adaptor.add_examples_to_dataset(dataset_name, inputs, outputs)
    fetched_examples = zen_arize_adaptor.fetch_dataset_examples(dataset_name)
    assert len(fetched_examples) == 2


@pytest.mark.helpers
def test_zen_arize_metric_evaluator(
    arize_phoenix: Client, zen_arize_adaptor: ZenArizeAdaptor, openai: OpenAI, test_set: str
):
    # GIVEN we have the zenbase manager and function
    zenbase_tracer = ZenbaseTracer()

    @zenbase_tracer
    def langchain_chain(request: LMRequest) -> str:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        messages = [
            (
                "system",
                "You are an expert math solver. Your answer must be just the number with no separators, and nothing else. Follow the format of the examples.",  # noqa
            )
        ]
        for demo in request.zenbase.task_demos:
            messages += [
                ("user", demo.inputs["question"]),
                ("assistant", demo.outputs["answer"]),
            ]

        messages.append(("user", "{question}"))

        chain = ChatPromptTemplate.from_messages(messages) | ChatOpenAI(model="gpt-3.5-turbo") | StrOutputParser()

        print("Mathing...")
        answer = chain.invoke(request.inputs["inputs"])
        return answer

    # GIVEN your langchain_chain function is defined and working
    return_langchain = langchain_chain({"inputs": {"question": "What is 2 + 2?"}})
    assert return_langchain is not None

    # GIVEN you have a function that scores the answer
    def score_answer(output: str, expected: dict):
        """The first argument is the return value from the `langchain_chain` function above."""
        score = int(output == expected["outputs"]["answer"].split("#### ")[-1])
        return score

    zen_arize_adaptor.set_evaluator_kwargs(
        evaluators=[score_answer],
    )
    evaluator = zen_arize_adaptor.get_evaluator(data=test_set)
    result = evaluator(langchain_chain)
    assert result.evals["score"] is not None
    assert len(result.individual_evals) != 0


@pytest.mark.helpers
def test_zen_arize_lcel_labeled_few_shot_learning(
    arize_phoenix: Client, zen_arize_adaptor: ZenArizeAdaptor, openai: OpenAI, test_set: str, optim: LabeledFewShot
):
    # GIVEN we have the zenbase manager and function
    zenbase_tracer = ZenbaseTracer()

    @zenbase_tracer
    def langchain_chain(request: LMRequest) -> str:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        messages = [
            (
                "system",
                "You are an expert math solver. Your answer must be just the number with no separators, and nothing else. Follow the format of the examples.",  # noqa
            )
        ]
        for demo in request.zenbase.task_demos:
            messages += [
                ("user", demo.inputs["question"]),
                ("assistant", demo.outputs["answer"]),
            ]

        messages.append(("user", "{question}"))

        chain = ChatPromptTemplate.from_messages(messages) | ChatOpenAI(model="gpt-3.5-turbo") | StrOutputParser()

        print("Mathing...")
        answer = chain.invoke(request.inputs["inputs"])
        return answer

    # GIVEN your langchain_chain function is defined and working
    return_langchain = langchain_chain({"inputs": {"question": "What is 2 + 2?"}})
    assert return_langchain is not None

    # GIVEN you have a function that scores the answer
    def score_answer(output: str, expected: dict):
        """The first argument is the return value from the `langchain_chain` function above."""
        # if there is any #### in the output
        if "####" in expected["outputs"]["answer"]:
            output = output.split("#### ")[-1]

        score = int(output == expected["outputs"]["answer"].split("#### ")[-1])
        return score

    # WHEN you optimize the function with the labeled few-shot learning
    fn, candidates, _ = optim.perform(
        langchain_chain,
        evaluator=ZenArizeAdaptor.metric_evaluator(
            dataset=arize_phoenix.get_dataset(name=test_set), evaluators=[score_answer]
        ),
        samples=SAMPLES,
        rounds=1,
    )

    # THEN the function should be optimized
    assert fn is not None
    assert any(candidates)
    assert next(e for e in candidates if 0 <= e.evals["score"] <= 1)


@pytest.mark.helpers
def test_zen_arize_lcel_multiple_calls(
    arize_phoenix: Client, zen_arize_adaptor: ZenArizeAdaptor, openai: OpenAI, test_set: str
):
    # GIVEN you have a function that scores the answer
    zenbase_tracer = ZenbaseTracer()

    @zenbase_tracer  # it is 1
    def solver(request: LMRequest):  # it is 2
        request.inputs = request.inputs["inputs"]
        messages = [
            (
                "system",
                """You are an expert math solver. Solve the given problem using the provided plan and operations.
            Return only the final numerical answer, without any additional text or explanation.""",
            ),
        ]

        for demo in request.zenbase.task_demos:  # it is 3
            messages += [
                ("user", f'Example Question: {demo.inputs["question"]}'),
                ("assistant", f'Example Answer: {demo.outputs["answer"]}'),
            ]  # it is 4

        messages.extend(
            [
                ("user", "Question: {question}"),
                ("user", "Plan: {plan}"),
                ("user", "Mathematical Operation: {operation}"),
                ("user", "Provide the final numerical answer:"),
            ]
        )

        chain = ChatPromptTemplate.from_messages(messages) | ChatOpenAI(model="gpt-3.5-turbo") | StrOutputParser()

        plan = planner_chain(request.inputs)
        operation = operation_finder({"plan": plan["plan"], "question": request.inputs["question"]})

        inputs_to_answer = {
            "question": request.inputs["question"],
            "plan": plan["plan"],
            "operation": operation["operation"],
        }
        answer = chain.invoke(inputs_to_answer)
        return {"answer": answer}

    @zenbase_tracer  # it is 1
    def planner_chain(request: LMRequest):  # it is 2
        messages = [
            (
                "system",
                """You are an expert math solver. Create a step-by-step plan to solve the given problem.
            Be clear and concise in your steps.""",
            ),
            ("user", "Problem: {question}\n\nProvide a step-by-step plan to solve this problem:"),
        ]

        if request.zenbase.task_demos:  # it is 3
            for demo in request.zenbase.task_demos[:2]:  # it is 4
                messages += [
                    ("user", demo.inputs["question"]),
                    ("assistant", demo.outputs["plan"]),
                ]

        chain = ChatPromptTemplate.from_messages(messages) | ChatOpenAI(model="gpt-3.5-turbo") | StrOutputParser()
        plan = chain.invoke(request.inputs)
        return {"plan": plan}

    @zenbase_tracer  # it is 1
    def operation_finder(request: LMRequest):  # it is 2
        messages = [
            (
                "system",
                """You are an expert math solver. Identify the overall mathematical operation needed to solve the
                problem
            based on the given plan. Use simple operations like addition, subtraction, multiplication, and division.""",
            ),
            ("user", "Question: {question}"),
            ("user", "Plan: {plan}"),
            ("user", "Identify the primary mathematical operation needed:"),
        ]

        if request.zenbase.task_demos:  # it is 3
            for demo in request.zenbase.task_demos[:2]:  # it is 4
                messages += [
                    ("user", demo.inputs["question"]),
                    ("user", demo.inputs["plan"]),
                    ("assistant", demo.outputs["operation"]),
                ]

        chain = ChatPromptTemplate.from_messages(messages) | ChatOpenAI(model="gpt-3.5-turbo") | StrOutputParser()
        operation = chain.invoke(request.inputs)
        return {"operation": operation}

    # GIVEN your langchain_chain function is defined and working
    return_langchain = solver({"inputs": {"question": "What is 2 + 2?"}})
    assert return_langchain is not None

    # GIVEN you have a function that scores the answer
    def score_answer(output: str, expected: dict):
        """The first argument is the return value from the `langchain_chain` function above."""

        score = int(output["answer"] == expected["outputs"]["answer"].split("#### ")[-1])
        return score

    zen_arize_adaptor.set_evaluator_kwargs(
        evaluators=[score_answer],
    )
    evaluator = zen_arize_adaptor.get_evaluator(data=test_set)
    result = evaluator(solver)
    assert result.evals["score"] is not None
    assert len(result.individual_evals) != 0


@pytest.mark.helpers
def test_zen_arize_lcel_bootstrap_few_shot(
    arize_phoenix: Client,
    zen_arize_adaptor: ZenArizeAdaptor,
    openai: OpenAI,
    test_set: str,
    train_set: str,
    validation_set: str,
):
    # GIVEN you have a function that scores the answer
    zenbase_tracer = ZenbaseTracer()

    @zenbase_tracer  # it is 1
    def solver(request: LMRequest):  # it is 2
        if "inputs" in request.inputs.keys():
            request.inputs = request.inputs["inputs"]
        else:
            pass
        messages = [
            (
                "system",
                """You are an expert math solver. Solve the given problem using the provided plan and operations.
            Return only the final numerical answer, without any additional text or explanation.""",
            ),
        ]

        for demo in request.zenbase.task_demos:  # it is 3
            demo_input = demo.inputs["inputs"]["question"]
            demo_output = demo.outputs["outputs"]["answer"]

            messages += [
                ("user", f"Example Question: {demo_input}"),
                ("assistant", f"Example Answer: {demo_output}"),
            ]  # it is 4

        messages.extend(
            [
                ("user", "Question: {question}"),
                ("user", "Plan: {plan}"),
                ("user", "Mathematical Operation: {operation}"),
                ("user", "Provide the final numerical answer:"),
            ]
        )

        chain = ChatPromptTemplate.from_messages(messages) | ChatOpenAI(model="gpt-3.5-turbo") | StrOutputParser()

        plan = planner_chain(request.inputs)
        operation = operation_finder({"plan": plan["plan"], "question": request.inputs["question"]})

        inputs_to_answer = {
            "question": request.inputs["question"],
            "plan": plan["plan"],
            "operation": operation["operation"],
        }
        answer = chain.invoke(inputs_to_answer)
        return {"answer": answer}

    @zenbase_tracer  # it is 1
    def planner_chain(request: LMRequest):  # it is 2
        messages = [
            (
                "system",
                """You are an expert math solver. Create a step-by-step plan to solve the given problem.
            Be clear and concise in your steps.""",
            ),
            ("user", "Problem: {question}\n\nProvide a step-by-step plan to solve this problem:"),
        ]

        if request.zenbase.task_demos:  # it is 3
            for demo in request.zenbase.task_demos[:2]:  # it is 4
                messages += [
                    ("user", demo.inputs["inputs"]["question"]),
                    ("assistant", demo.outputs["outputs"]["plan"]),
                ]

        chain = ChatPromptTemplate.from_messages(messages) | ChatOpenAI(model="gpt-3.5-turbo") | StrOutputParser()
        plan = chain.invoke(request.inputs)
        return {"plan": plan}

    @zenbase_tracer  # it is 1
    def operation_finder(request: LMRequest):  # it is 2
        messages = [
            (
                "system",
                """You are an expert math solver. Identify the overall mathematical operation needed to solve the
                 problem
            based on the given plan. Use simple operations like addition, subtraction, multiplication, and division.""",
            ),
            ("user", "Question: {question}"),
            ("user", "Plan: {plan}"),
            ("user", "Identify the primary mathematical operation needed:"),
        ]

        if request.zenbase.task_demos:  # it is 3
            for demo in request.zenbase.task_demos[:2]:  # it is 4
                messages += [
                    ("user", demo.inputs["inputs"]["question"]),
                    ("user", demo.inputs["inputs"]["plan"]),
                    ("assistant", demo.outputs["outputs"]["operation"]),
                ]

        chain = ChatPromptTemplate.from_messages(messages) | ChatOpenAI(model="gpt-3.5-turbo") | StrOutputParser()
        operation = chain.invoke(request.inputs)
        return {"operation": operation}

    # GIVEN your langchain_chain function is defined and working
    return_langchain = solver({"inputs": {"question": "What is 2 + 2?"}})
    assert return_langchain is not None

    # GIVEN you have a function that scores the answer
    def score_answer(output: str, expected: dict):
        """The first argument is the return value from the `langchain_chain` function above."""
        try:
            score = int(output["answer"] == expected["outputs"]["answer"].split("#### ")[-1])
        except Exception as e:
            raise e
        return score

    bootstrap_few_shot = BootstrapFewShot(
        shots=SHOTS,
        training_set=train_set,
        test_set=test_set,
        validation_set=validation_set,
        evaluator_kwargs=dict(
            evaluators=[score_answer],
        ),
        zen_adaptor=zen_arize_adaptor,
    )
    teacher_lm, candidates = bootstrap_few_shot.perform(
        solver,
        samples=SAMPLES,
        rounds=1,
        trace_manager=zenbase_tracer,
    )
    assert teacher_lm is not None

    zenbase_tracer.all_traces = {}
    teacher_lm({"inputs": {"question": "What is 2 + 2?"}})

    assert [v for k, v in zenbase_tracer.all_traces.items()][0]["optimized"]["planner_chain"]["args"][
        "request"
    ].zenbase.task_demos[0].inputs is not None
    assert [v for k, v in zenbase_tracer.all_traces.items()][0]["optimized"]["operation_finder"]["args"][
        "request"
    ].zenbase.task_demos[0].inputs is not None
    assert [v for k, v in zenbase_tracer.all_traces.items()][0]["optimized"]["solver"]["args"][
        "request"
    ].zenbase.task_demos[0].inputs is not None

    path_of_the_file = os.path.join(TEST_DIR, "adaptors/bootstrap_few_shot_optimizer_test.zenbase")

    bootstrap_few_shot.save_optimizer_args(path_of_the_file)

    # assert that the file has been saved
    assert os.path.exists(path_of_the_file)
