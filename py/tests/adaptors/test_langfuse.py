import logging
import os

import pytest
from datasets import DatasetDict
from langfuse import Langfuse
from langfuse.decorators import observe
from openai import OpenAI
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
)

from zenbase.adaptors.langfuse_helper import ZenLangfuse
from zenbase.core.managers import ZenbaseTracer
from zenbase.optim.metric.bootstrap_few_shot import BootstrapFewShot
from zenbase.optim.metric.labeled_few_shot import LabeledFewShot
from zenbase.optim.metric.types import OverallEvalValue
from zenbase.settings import TEST_DIR
from zenbase.types import LMDemo, LMRequest
from zenbase.utils import ksuid, pmap

SAMPLES = 2
SHOTS = 3
EVALSET_SIZE = 5
TESTSET_SIZE = 2
TRAINSET_SIZE = 5
VALIDATIONSET_SIZE = 2

log = logging.getLogger(__name__)


@pytest.fixture
def optim(gsm8k_demoset: list):
    return LabeledFewShot(demoset=gsm8k_demoset, shots=SHOTS)


@pytest.fixture(scope="module")
def langfuse():
    client = Langfuse()
    client.auth_check()
    return client


@pytest.fixture(scope="module")
def openai():
    return OpenAI()


@pytest.fixture(scope="module")
def zen_langfuse_helper(langfuse: Langfuse):
    return ZenLangfuse(langfuse)


def create_dataset_with_examples(zen_langfuse_helper: ZenLangfuse, prefix: str, item_set: list):
    dataset_name = ksuid(prefix=prefix)

    zen_langfuse_helper.create_dataset(dataset_name)
    inputs = [{"question": example["question"]} for example in item_set]
    expected_outputs = [example["answer"] for example in item_set]
    zen_langfuse_helper.add_examples_to_dataset(dataset_name, inputs, expected_outputs)
    return dataset_name


@pytest.fixture(scope="module")
def train_set(gsm8k_dataset: DatasetDict, zen_langfuse_helper: ZenLangfuse):
    return create_dataset_with_examples(
        zen_langfuse_helper,
        "GSM8K_train_set_langsmith_dataset",
        list(gsm8k_dataset["train"].select(range(TRAINSET_SIZE))),
    )


@pytest.fixture(scope="module")
def validation_set(gsm8k_dataset: DatasetDict, zen_langfuse_helper: ZenLangfuse):
    return create_dataset_with_examples(
        zen_langfuse_helper,
        "GSM8K_validation_set_langsmith_dataset",
        list(gsm8k_dataset["train"].select(range(TRAINSET_SIZE + 1, TRAINSET_SIZE + VALIDATIONSET_SIZE + 1))),
    )


@pytest.fixture(scope="module")
def test_set(gsm8k_dataset: DatasetDict, zen_langfuse_helper: ZenLangfuse):
    return create_dataset_with_examples(
        zen_langfuse_helper,
        "GSM8K_test_set_langsmith_dataset",
        list(gsm8k_dataset["test"].select(range(TESTSET_SIZE))),
    )


@pytest.fixture(scope="module")
def evalset(gsm8k_dataset: DatasetDict, langfuse: Langfuse):
    try:
        return langfuse.get_dataset("gsm8k-testset")
    except:  # noqa: E722
        langfuse.create_dataset("gsm8k-testset")
        pmap(
            lambda example: langfuse.create_dataset_item(
                dataset_name="gsm8k-testset",
                input={"question": example["question"]},
                expected_output=example["answer"],
            ),
            gsm8k_dataset["test"].select(range(EVALSET_SIZE)),
        )
        return langfuse.get_dataset("gsm8k-testset")


def score_answer(answer: str, demo: LMDemo, langfuse: Langfuse) -> OverallEvalValue:
    """The first argument is the return value from the `langchain_chain` function above."""
    score = int(answer.split("#### ")[-1] == demo.outputs.split("#### ")[-1])
    langfuse.score(
        name="correctness",
        value=score,
        trace_id=langfuse.get_trace_id(),
    )
    return {"score": score}


@pytest.mark.helpers
def test_langfuse_lcel_labeled_few_shot(optim: LabeledFewShot, evalset: list):
    trace_manager = ZenbaseTracer()

    @trace_manager.trace_function
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(max=8),
        before_sleep=before_sleep_log(log, logging.WARN),
    )
    @observe()
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
        answer = chain.invoke(request.inputs)
        return answer

    fn, candidates, _ = optim.perform(
        langchain_chain,
        evaluator=ZenLangfuse.metric_evaluator(evalset, evaluate=score_answer),
        samples=SAMPLES,
        rounds=1,
    )

    assert fn is not None
    assert any(candidates)
    assert next(e for e in candidates if 0.5 <= e.evals["score"] <= 1)


@pytest.mark.helpers
def test_zen_langfuse_metric_evaluator(langfuse: Langfuse, zen_langfuse_helper: ZenLangfuse, evalset: list):
    zenbase_manager = ZenbaseTracer()

    @zenbase_manager.trace_function
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(max=8),
        before_sleep=before_sleep_log(log, logging.WARN),
    )
    @observe()
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
        answer = chain.invoke(request.inputs)
        return answer

    langchain_chain({"question": "What is 2 + 2?"})
    zen_langfuse_helper.set_evaluator_kwargs(
        evaluate=score_answer,
    )
    evaluator = zen_langfuse_helper.get_evaluator(data="gsm8k-testset")
    result = evaluator(langchain_chain)
    assert result.evals["score"] is not None
    assert len(result.individual_evals) != 0


@pytest.mark.helpers
def test_bootstrap_few_shot_langfuse(
    train_set,
    validation_set,
    test_set,
    zen_langfuse_helper: ZenLangfuse,
):
    zenbase_manager = ZenbaseTracer()

    def score_answer_with_json(answer: str, demo: LMDemo, langfuse: Langfuse) -> OverallEvalValue:
        """The first argument is the return value from the `langchain_chain` function above."""
        score = int(answer["answer"] == demo.outputs.split("#### ")[-1])
        langfuse.score(
            name="correctness",
            value=score,
            trace_id=langfuse.get_trace_id(),
        )
        return {"score": score}

    @zenbase_manager.trace_function
    # @retry(
    #     stop=stop_after_attempt(3),
    #     wait=wait_exponential_jitter(max=8),
    #     before_sleep=before_sleep_log(log, logging.WARN),
    # )
    @observe()
    def solver(request: LMRequest):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        messages = [
            (
                "system",
                "You are an expert math solver. "
                "You have an question that you should answer, "
                "You have step by step actions that you should take to solve the problem."
                "You have the opertaions that you should do to solve the problem"
                "You should come just with the number for the answer, just the actual number like examples that you have."  # noqa
                ""
                ""
                "Follow the format of the examples as they have the final answer, you need to came up to the plan for solving them.",  # noqa
            )
        ]
        for demo in request.zenbase.task_demos:
            if isinstance(demo.outputs, dict):
                the_output = demo.outputs["answer"]
            else:
                the_output = demo.outputs

            messages += [
                ("user", f'Example Question: {demo.inputs["question"]}'),
                ("assistant", f"Example Answer: {the_output}"),
            ]

        messages.append(("user", "Question: {question}"))
        messages.append(("user", "Plan: {plan}"))
        messages.append(("user", "Mathematical Operation that needed: {operation}"))
        messages.append(
            ("user", "Now come with the answer as number, just return the number, nothing else, just NUMBERS.")
        )

        chain = ChatPromptTemplate.from_messages(messages) | ChatOpenAI(model="gpt-3.5-turbo") | StrOutputParser()

        print("Mathing...")
        plan = planner_chain(request.inputs)
        the_plan = plan["plan"]
        the_operation = operation_finder(
            {
                "plan": the_plan,
                "question": request.inputs["question"],
            }
        )
        inputs_to_answer = {
            "question": request.inputs["question"],
            "plan": the_plan,
            "operation": the_operation["operation"],
        }
        answer = chain.invoke(inputs_to_answer)
        return {"answer": answer}

    @zenbase_manager.trace_function
    # @retry(
    #     stop=stop_after_attempt(3),
    #     wait=wait_exponential_jitter(max=8),
    #     before_sleep=before_sleep_log(log, logging.WARN),
    # )
    @observe()
    def planner_chain(request: LMRequest):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        messages = [
            (
                "system",
                "You are an expert math solver. You have an question that you should create step-by-step plan to solve it. "  # noqa
                "Follow the format of the examples.",
                # noqa
            )
        ]
        if request.zenbase.task_demos:
            for demo in request.zenbase.task_demos[:2]:
                messages += [
                    ("user", demo.inputs["question"]),
                    ("assistant", demo.outputs["plan"]),
                ]

        messages.append(("user", "{question}"))

        chain = ChatPromptTemplate.from_messages(messages) | ChatOpenAI(model="gpt-3.5-turbo") | StrOutputParser()

        print("Mathing...")
        answer = chain.invoke(request.inputs)
        return {"plan": answer}

    @zenbase_manager.trace_function
    # @retry(
    #     stop=stop_after_attempt(3),
    #     wait=wait_exponential_jitter(max=8),
    #     before_sleep=before_sleep_log(log, logging.WARN),
    # )
    @observe()
    def operation_finder(request: LMRequest):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        messages = [
            (
                "system",
                "You are an expert math solver. You have a plan for solve a problem that is step-by-step, you need to find the overal operation in the math to solve it. "  # noqa
                "Just come up with math operation with simple match operations like sum, multiply, division and minus. "
                ""
                "Follow the format of the examples.",
                # noqa
            )
        ]
        if request.zenbase.task_demos:
            for demo in request.zenbase.task_demos[:2]:
                messages += [
                    ("user", demo.inputs["question"]),
                    ("user", demo.inputs["plan"]),
                    ("assistant", demo.outputs["operation"]),
                ]

        messages.append(("user", "{question}"))
        messages.append(("user", "{plan}"))

        chain = ChatPromptTemplate.from_messages(messages) | ChatOpenAI(model="gpt-3.5-turbo") | StrOutputParser()

        print("Mathing...")
        answer = chain.invoke(request.inputs)
        return {"operation": answer}

    evaluator_kwargs = dict(
        evaluate=score_answer_with_json,
    )

    solver({"question": "2+2?"})

    bootstrap_few_shot = BootstrapFewShot(
        shots=SHOTS,
        training_set=train_set,
        test_set=test_set,
        validation_set=validation_set,
        evaluator_kwargs=evaluator_kwargs,
        zen_adaptor=zen_langfuse_helper,
    )

    teacher_lm, candidates = bootstrap_few_shot.perform(
        solver,
        samples=SAMPLES,
        rounds=1,
        trace_manager=zenbase_manager,
    )

    assert teacher_lm is not None

    zenbase_manager.all_traces = {}
    teacher_lm({"question": "What is 2 + 2?"})

    assert [v for k, v in zenbase_manager.all_traces.items()][0]["optimized"]["planner_chain"]["args"][
        "request"
    ].zenbase.task_demos[0].inputs is not None
    assert [v for k, v in zenbase_manager.all_traces.items()][0]["optimized"]["operation_finder"]["args"][
        "request"
    ].zenbase.task_demos[0].inputs is not None
    assert [v for k, v in zenbase_manager.all_traces.items()][0]["optimized"]["solver"]["args"][
        "request"
    ].zenbase.task_demos[0].inputs is not None

    path_of_the_file = os.path.join(TEST_DIR, "adaptors/bootstrap_few_shot_optimizer_test.zenbase")

    bootstrap_few_shot.save_optimizer_args(path_of_the_file)

    # assert that the file has been saved
    assert os.path.exists(path_of_the_file)
