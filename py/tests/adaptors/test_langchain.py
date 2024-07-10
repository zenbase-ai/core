import json
import logging
import os
from typing import TYPE_CHECKING

import pytest
from datasets import DatasetDict
from langsmith import Client, traceable
from langsmith import utils as ls_utils
from langsmith.schemas import Example, Run
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
)

from zenbase.adaptors.langchain import ZenLangSmith
from zenbase.core.managers import ZenbaseTracer
from zenbase.optim.metric.bootstrap_few_shot import BootstrapFewShot
from zenbase.optim.metric.labeled_few_shot import LabeledFewShot
from zenbase.settings import TEST_DIR
from zenbase.types import LMDemo, LMRequest
from zenbase.utils import ksuid

if TYPE_CHECKING:
    from langsmith import schemas

SAMPLES = 2
SHOTS = 2
TESTSET_SIZE = 2
TRAINSET_SIZE = 5
VALIDATIONSET_SIZE = 2

log = logging.getLogger(__name__)


@pytest.fixture
def labeled_few_shot_optimizer(gsm8k_demoset: list):
    return LabeledFewShot(demoset=gsm8k_demoset, shots=SHOTS)


@pytest.fixture(scope="module")
def openai():
    return wrap_openai(
        OpenAI(
            # base_url="http://0.0.0.0:4000",
        )
    )


@pytest.fixture(scope="module")
def chat_openai():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="gpt-3.5-turbo")


@pytest.fixture(scope="module")
def langsmith():
    return Client()


@pytest.fixture(scope="module")
def langsmith_helper(langsmith):
    return ZenLangSmith(client=langsmith)


def create_dataset_with_examples(
    smith_helper: ZenLangSmith, prefix: str, description: str, item_set: list
) -> "schemas.Dataset":
    # Generate dataset name
    dataset_name = ksuid(prefix=prefix)

    # Create dataset
    dataset = smith_helper.create_dataset(dataset_name, description)
    inputs = [{"question": example["question"]} for example in item_set]
    outputs = [{"answer": example["answer"]} for example in item_set]
    smith_helper.add_examples_to_dataset(dataset.name, inputs, outputs)
    return dataset


@pytest.fixture(scope="module")
def train_set(gsm8k_dataset: DatasetDict, langsmith_helper: ZenLangSmith):
    return create_dataset_with_examples(
        langsmith_helper,
        "GSM8K_train_set_langsmith_dataset",
        "GSM8K math reasoning dataset",
        list(gsm8k_dataset["train"].select(range(TRAINSET_SIZE))),
    )


@pytest.fixture(scope="module")
def validation_set(gsm8k_dataset: DatasetDict, langsmith_helper: ZenLangSmith):
    return create_dataset_with_examples(
        langsmith_helper,
        "GSM8K_validation_set_langsmith_dataset",
        "GSM8K math reasoning dataset",
        list(gsm8k_dataset["train"].select(range(TRAINSET_SIZE + 1, TRAINSET_SIZE + VALIDATIONSET_SIZE + 1))),
    )


@pytest.fixture(scope="module")
def test_set(gsm8k_dataset: DatasetDict, langsmith_helper: ZenLangSmith):
    return create_dataset_with_examples(
        langsmith_helper,
        "GSM8K_test_set_langsmith_dataset",
        "GSM8K math reasoning dataset",
        list(gsm8k_dataset["test"].select(range(TESTSET_SIZE))),
    )


@pytest.fixture(scope="module")
def evalset(gsm8k_dataset: DatasetDict, langsmith: Client):
    try:
        return list(langsmith.list_examples(dataset_name="gsm8k-test-examples"))
    except ls_utils.LangSmithNotFoundError:
        dataset = langsmith.create_dataset("gsm8k-test-examples")
        examples = gsm8k_dataset["test"].select(range(TESTSET_SIZE))
        langsmith.create_examples(
            inputs=[{"question": e["question"]} for e in examples],
            outputs=[{"answer": e["answer"]} for e in examples],
            dataset_id=dataset.id,
        )
        return list(langsmith.list_examples(dataset_name="gsm8k-test-examples"))


def score_answer(run: Run, example: Example):
    output = run.outputs["answer"].split("#### ")[-1]
    target = example.outputs["answer"].split("#### ")[-1]
    return {
        "key": "correctness",
        "score": int(output == target),
    }


@pytest.mark.helpers
def test_zenlanchain_metric_evaluator(
    langsmith: Client,
    evalset: list,
):
    zenbase_manager = ZenbaseTracer()

    @zenbase_manager.trace_function
    @traceable
    def langchain_chain(request: LMRequest):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        messages = [
            (
                "system",
                "You are an expert math solver. Your answer must be just the number with no separators, and nothing "
                "else. Follow the format of the examples.",
                # noqa
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
        return {"answer": answer}

    langchain_chain({"question": "What is 2 + 2?"})

    zenbase_evaluator = ZenLangSmith.metric_evaluator(
        data=evalset,
        evaluators=[score_answer],
        client=langsmith,
        max_concurrency=2,
    )
    result = zenbase_evaluator(langchain_chain)
    assert result.evals["score"] is not None


@pytest.mark.helpers
def test_langsmith_lcel_labeled_few_shot(
    langsmith: Client,
    labeled_few_shot_optimizer: LabeledFewShot,
    evalset: list,
):
    trace_manager = ZenbaseTracer()

    @trace_manager.trace_function
    # @retry(
    #     stop=stop_after_attempt(3),
    #     wait=wait_exponential_jitter(max=8),
    #     before_sleep=before_sleep_log(log, logging.WARN),
    # )
    @traceable
    def langchain_chain(request: LMRequest):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        langchain_chain_2(request)

        messages = [
            (
                "system",
                "You are an expert math solver. Your answer must be just the number with no separators, and nothing "
                "else. Follow the format of the examples.",
                # noqa
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
        return {"answer": answer}

    @trace_manager.trace_function
    # @retry(
    #     stop=stop_after_attempt(3),
    #     wait=wait_exponential_jitter(max=8),
    #     before_sleep=before_sleep_log(log, logging.WARN),
    # )
    @traceable
    def langchain_chain_2(request: LMRequest):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        messages = [
            (
                "system",
                "You are an expert math solver. Your answer must be just the number with no separators, and nothing "
                "else. Follow the format of the examples.",
                # noqa
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
        return {"answer": answer}

    fn, candidates, best_candidate_result = labeled_few_shot_optimizer.perform(
        langchain_chain,
        evaluator=ZenLangSmith.metric_evaluator(
            data=evalset,
            evaluators=[score_answer],
            client=langsmith,
            max_concurrency=2,
        ),
        samples=SAMPLES,
        rounds=1,
    )

    assert fn is not None
    assert any(candidates)
    assert next(c for c in candidates if 0.5 <= c.evals["score"] <= 1)


@pytest.mark.helpers
def test_create_dataset(langsmith_helper):
    dataset_name = ksuid("test_dataset_creation")
    description = "Test description"
    dataset = langsmith_helper.create_dataset(dataset_name, description)
    assert dataset.name == dataset_name
    assert dataset.description == description


@pytest.mark.helpers
def test_add_examples_to_dataset(langsmith_helper):
    dataset_name = ksuid("test_dataset_examples")
    description = "Test description for adding examples"
    dataset = langsmith_helper.create_dataset(dataset_name, description)
    inputs = [{"question": "Q1"}]
    outputs = [{"answer": "A1"}]
    langsmith_helper.add_examples_to_dataset(dataset.name, inputs, outputs)
    dataset_examples = langsmith_helper.fetch_dataset_examples(dataset_name)
    dataset_data = [e for e in dataset_examples]
    assert len(dataset_data) > 0


@pytest.mark.helpers
def test_fetch_dataset(langsmith_helper):
    dataset_name = ksuid("test_dataset_fetch")
    description = "Test description for fetch"
    dataset = langsmith_helper.create_dataset(dataset_name, description)
    fetched_dataset = langsmith_helper.fetch_dataset(dataset.name)
    assert fetched_dataset.name == dataset_name


@pytest.mark.helpers
def test_fetch_dataset_demos(langsmith_helper):
    dataset_name = ksuid("test_dataset_demos")
    description = "Test description for demos"
    dataset = langsmith_helper.create_dataset(dataset_name, description)
    inputs = [{"question": "Q1"}]
    outputs = [{"answer": "A1"}]
    langsmith_helper.add_examples_to_dataset(dataset.name, inputs, outputs)
    demos = langsmith_helper.fetch_dataset_demos(dataset)

    assert isinstance(demos[0], LMDemo)
    assert len(demos) > 0
    assert demos[0].inputs["question"] == "Q1"
    assert demos[0].outputs["answer"] == "A1"


@pytest.mark.helpers
def test_bootstrap_few_shot_langchain_load_args(
    train_set: "schemas.Dataset",
    validation_set: "schemas.Dataset",
    test_set: "schemas.Dataset",
    langsmith_helper: ZenLangSmith,
):
    zenbase_manager = ZenbaseTracer()

    @zenbase_manager.trace_function
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(max=8),
        before_sleep=before_sleep_log(log, logging.WARN),
    )
    @traceable
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
                # noqa
                # noqa
            )
        ]
        for demo in request.zenbase.task_demos:
            messages += [
                ("user", f'Example Question: {demo.inputs["question"]}'),
                ("assistant", f'Example Answer: {demo.outputs["answer"]}'),
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
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(max=8),
        before_sleep=before_sleep_log(log, logging.WARN),
    )
    @traceable
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
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(max=8),
        before_sleep=before_sleep_log(log, logging.WARN),
    )
    @traceable
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

    path_of_the_file = os.path.join(TEST_DIR, "adaptors/bootstrap_few_shot_optimizer_args.zenbase")

    teacher_lm = BootstrapFewShot.load_optimizer_and_function(
        optimizer_args_file=path_of_the_file, student_lm=solver, trace_manager=zenbase_manager
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


@pytest.mark.helpers
def test_bootstrap_few_shot_langchain(
    train_set: "schemas.Dataset",
    validation_set: "schemas.Dataset",
    test_set: "schemas.Dataset",
    langsmith_helper: ZenLangSmith,
):
    zenbase_manager = ZenbaseTracer()

    @zenbase_manager
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(max=8),
        before_sleep=before_sleep_log(log, logging.WARN),
    )
    @traceable
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
                # noqa
                # noqa
            )
        ]
        for demo in request.zenbase.task_demos:
            messages += [
                ("user", f'Example Question: {demo.inputs["question"]}'),
                ("assistant", f'Example Answer: {demo.outputs["answer"]}'),
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

    @zenbase_manager
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(max=8),
        before_sleep=before_sleep_log(log, logging.WARN),
    )
    @traceable
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

    @zenbase_manager
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(max=8),
        before_sleep=before_sleep_log(log, logging.WARN),
    )
    @traceable
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
        evaluators=[score_answer],
        client=langsmith_helper.client,
        max_concurrency=1,
    )

    bootstrap_few_shot = BootstrapFewShot(
        shots=SHOTS,
        training_set=train_set,
        test_set=test_set,
        validation_set=validation_set,
        evaluator_kwargs=evaluator_kwargs,
        zen_adaptor=langsmith_helper,
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


@pytest.mark.helpers
def test_bootstrap_few_shot_openai_langsmith(
    train_set: "schemas.Dataset",
    validation_set: "schemas.Dataset",
    test_set: "schemas.Dataset",
    langsmith_helper: ZenLangSmith,
    openai: OpenAI,
):
    zenbase_manager = ZenbaseTracer()

    @zenbase_manager.trace_function
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(max=8),
        before_sleep=before_sleep_log(log, logging.WARN),
    )
    @traceable
    def solver(request: LMRequest):
        messages = [
            {
                "role": "system",
                "content": "You are an expert math solver. You have a question that you should answer. You have step by step actions that you should take to solve the problem. You have the operations that you should do to solve the problem. You should come just with the number for the answer, just the actual number like examples that you have. Follow the format of the examples as they have the final answer, you need to come up with the plan for solving them."  # noqa
                'return it with json like return it in the {"answer": " the answer "}',
            }
        ]

        for demo in request.zenbase.task_demos:
            messages += [
                {"role": "user", "content": f"Example Question: {demo.inputs['question']}"},
                {"role": "assistant", "content": f"Example Answer: {demo.outputs['answer']}"},
            ]

        plan = planner_chain(request.inputs)
        the_plan = plan["plan"]
        the_operation = operation_finder(
            {
                "plan": the_plan,
                "question": request.inputs["question"],
            }
        )

        messages.append({"role": "user", "content": f"Question: {request.inputs['question']}"})
        messages.append({"role": "user", "content": f"Plan: {the_plan}"})
        messages.append(
            {"role": "user", "content": f"Mathematical Operation that needed: {the_operation['operation']}"}
        )
        messages.append(
            {
                "role": "user",
                "content": "Now come with the answer as number, just return the number, nothing else, just NUMBERS.",
            }
        )

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            response_format={"type": "json_object"},
        )

        print("Mathing...")
        answer = json.loads(response.choices[0].message.content)
        return {"answer": answer["answer"]}

    @zenbase_manager.trace_function
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(max=8),
        before_sleep=before_sleep_log(log, logging.WARN),
    )
    @traceable
    def planner_chain(request: LMRequest):
        messages = [
            {
                "role": "system",
                "content": "You are an expert math solver. You have a question that you should create a step-by-step plan to solve it. Follow the format of the examples and return JSON object."  # noqa
                'return it in the {"plan": " the plan "}',
            }
        ]

        if request.zenbase.task_demos:
            for demo in request.zenbase.task_demos[:2]:
                messages += [
                    {"role": "user", "content": demo.inputs["question"]},
                    {"role": "assistant", "content": demo.outputs["plan"]},
                ]

        messages.append({"role": "user", "content": request.inputs["question"]})

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            response_format={"type": "json_object"},
        )

        print("Planning...")
        answer = json.loads(response.choices[0].message.content)
        return {"plan": " ".join(i for i in answer["plan"])}

    @zenbase_manager.trace_function
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(max=8),
        before_sleep=before_sleep_log(log, logging.WARN),
    )
    @traceable
    def operation_finder(request: LMRequest):
        messages = [
            {
                "role": "system",
                "content": "You are an expert math solver. You have a plan for solving a problem that is step-by-step, you need to find the overall operation in the math to solve it. Just come up with math operation with simple math operations like sum, multiply, division and minus. Follow the format of the examples."  # noqa
                'return it with json like return it in the {"operation": " the operation "}',
            }
        ]

        if request.zenbase.task_demos:
            for demo in request.zenbase.task_demos[:2]:
                messages += [
                    {"role": "user", "content": f"Question: {demo.inputs['question']}"},
                    {"role": "user", "content": f"Plan: {demo.inputs['plan']}"},
                    {"role": "assistant", "content": demo.outputs["operation"]},
                ]

        messages.append({"role": "user", "content": f"Question: {request.inputs['question']}"})
        messages.append({"role": "user", "content": f"Plan: {request.inputs['plan']}"})

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            response_format={"type": "json_object"},
        )

        print("Finding operation...")
        answer = json.loads(response.choices[0].message.content)
        return {"operation": answer["operation"]}

    evaluator_kwargs = dict(
        evaluators=[score_answer],
        client=langsmith_helper.client,
        max_concurrency=1,
    )

    bootstrap_few_shot = BootstrapFewShot(
        shots=SHOTS,
        training_set=train_set,
        test_set=test_set,
        validation_set=validation_set,
        evaluator_kwargs=evaluator_kwargs,
        zen_adaptor=langsmith_helper,
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

    path_of_the_file = os.path.join(TEST_DIR, "adaptors/bootstrap_fewshot_output_test.zenbase")

    bootstrap_few_shot.save_optimizer_args(path_of_the_file)

    # assert that the file has been saved
    assert os.path.exists(path_of_the_file)
