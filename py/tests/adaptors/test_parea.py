import json
import logging
import os

import pytest
from datasets import DatasetDict
from openai import OpenAI
from parea import Parea, trace
from parea.schemas import EvaluationResult, Log
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
)

from zenbase.adaptors.parea import ZenParea
from zenbase.core.managers import ZenbaseTracer
from zenbase.optim.metric.bootstrap_few_shot import BootstrapFewShot
from zenbase.optim.metric.labeled_few_shot import LabeledFewShot
from zenbase.settings import TEST_DIR
from zenbase.types import LMRequest, deflm
from zenbase.utils import expand_nested_json, ksuid

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
def parea():
    return Parea()


@pytest.fixture(scope="module")
def openai(parea: Parea):
    client = OpenAI()
    parea.wrap_openai_client(client)
    return client


@pytest.fixture(scope="module")
def chat_openai(parea: Parea):
    from langchain_openai import ChatOpenAI

    client = ChatOpenAI(model="gpt-3.5-turbo", temperature=1, top_p=1, frequency_penalty=0, presence_penalty=0)
    return client


@pytest.fixture(scope="module")
def evalset(gsm8k_dataset: DatasetDict, parea: Parea):
    try:
        return [
            {"inputs": json.loads(case.inputs["inputs"]), "target": case.target}
            for case in parea.get_collection("gsm8k-testset").test_cases.values()
        ]
    except TypeError:
        data = [
            {"inputs": {"question": example["question"]}, "target": example["answer"]}
            for example in gsm8k_dataset["test"].select(range(EVALSET_SIZE))
        ]
        parea.create_test_collection(data, name="gsm8k-testset")
        return data


TEST_SET_SIZE = 5


@pytest.fixture(scope="module")
def zen_parea_helper(parea: Parea):
    return ZenParea(parea)


@pytest.mark.helpers
def test_create_and_fetch_dataset(parea: Parea, zen_parea_helper: ZenParea):
    dataset_name = str(ksuid("gsm8k-test"))
    zen_parea_helper.create_dataset(dataset_name)
    fetched_dataset = zen_parea_helper.fetch_dataset(dataset_name)
    assert fetched_dataset is not None


@pytest.mark.helpers
def test_create_and_add_examples(parea: Parea, zen_parea_helper: ZenParea):
    inputs = [{"question": "What is 1+1?"}, {"question": "What is 2+2?"}]
    outputs = [{"answer:": "2"}, {"answer": "4"}]
    dataset_name = str(ksuid("gsm8k-test"))
    zen_parea_helper.create_dataset(dataset_name)
    zen_parea_helper.add_examples_to_dataset(inputs, outputs, dataset_name)
    fetched_examples = zen_parea_helper.fetch_dataset_examples(dataset_name)
    assert len(fetched_examples) == 2


@pytest.mark.helpers
def create_dataset_with_examples(zen_parea_helper: ZenParea, prefix: str, item_set):
    dataset_name = ksuid(prefix)
    zen_parea_helper.create_dataset(dataset_name)
    inputs = [{"question": example["question"]} for example in item_set]
    outputs = [example["answer"] for example in item_set]
    zen_parea_helper.add_examples_to_dataset(inputs, outputs, dataset_name)
    return dataset_name


@pytest.fixture(scope="module")
def train_set(gsm8k_dataset: DatasetDict, zen_parea_helper: ZenParea):
    return create_dataset_with_examples(
        zen_parea_helper,
        "GSM8K_train_set_parea_dataset",
        list(gsm8k_dataset["train"].select(range(TRAINSET_SIZE))),
    )


@pytest.fixture(scope="module")
def test_set(gsm8k_dataset: DatasetDict, zen_parea_helper: ZenParea):
    return create_dataset_with_examples(
        zen_parea_helper,
        "GSM8K_test_set_parea_dataset",
        list(gsm8k_dataset["train"].select(range(TEST_SET_SIZE))),
    )


@pytest.fixture(scope="module")
def validation_set(gsm8k_dataset: DatasetDict, zen_parea_helper: ZenParea):
    return create_dataset_with_examples(
        zen_parea_helper,
        "GSM8K_validation_set_parea_dataset",
        list(gsm8k_dataset["test"].select(range(TRAINSET_SIZE + 1, TRAINSET_SIZE + VALIDATIONSET_SIZE + 1))),
    )


def score_answer(log: Log) -> EvaluationResult:
    output = log.output.split("#### ")[-1]
    target = log.target.split("#### ")[-1]
    return EvaluationResult("correctness", int(output == target))


@pytest.mark.helpers
def test_zen_parea_helper_evaluator(parea: Parea, evalset: list, chat_openai):
    zenbase_manager = ZenbaseTracer()

    @zenbase_manager.trace_function
    @trace(eval_funcs=[score_answer])
    def langchain_chain(request: LMRequest):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

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

        chain = ChatPromptTemplate.from_messages(messages) | chat_openai | StrOutputParser()

        print("Mathing...")
        answer = chain.invoke(request.inputs)
        return {"answer": answer}

    langchain_chain({"question": "What is 2 + 2?"})

    zenbase_evaluator = ZenParea.metric_evaluator(data=evalset, n_workers=1, p=parea)
    result = zenbase_evaluator(langchain_chain)
    assert result.evals["score"] is not None


@pytest.mark.helpers
def test_zen_parea_helper_get_evaluator(parea: Parea, test_set, zen_parea_helper: ZenParea, openai):
    zenbase_manager = ZenbaseTracer()

    def score_answer_with_json(log: Log) -> EvaluationResult:
        output = str(expand_nested_json(log.output)["answer"])
        target = log.target.split("#### ")[-1]
        return EvaluationResult("correctness", int(output == target))

    @zenbase_manager.trace_function
    @trace(eval_funcs=[score_answer_with_json])
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

        messages.append(
            {
                "role": "user",
                "content": "Now come with the answer as number, just return the number, nothing else, just NUMBERS.",
            }
        )
        messages.append({"role": "user", "content": request.inputs["question"]})
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        print("Mathing...")
        answer = json.loads(response.choices[0].message.content)
        return {"answer": answer["answer"]}

    zen_parea_helper.set_evaluator_kwargs(p=parea, n_workers=1)

    zenbase_evaluator = zen_parea_helper.get_evaluator(data=test_set)
    result = zenbase_evaluator(solver)
    assert result.evals["score"] is not None


@pytest.mark.helpers
def test_parea_lcel_labeled_few_shot(
    optim: LabeledFewShot,
    parea: Parea,
    evalset: list,
):
    @deflm
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(max=8),
        before_sleep=before_sleep_log(log, logging.WARN),
    )
    @trace(eval_funcs=[score_answer])
    def langchain_chain(request: LMRequest):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        messages = [  # noqa
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

        chain_2 = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are the verifier",
                    ),
                    (
                        "user",
                        "the question is: {question} and the answer is {answer}, is it right? answer with yes or no.",
                    ),
                ]
            )
            | ChatOpenAI(model="gpt-3.5-turbo")
            | StrOutputParser()
        )

        print("Mathing...")
        new_answer = chain_2.invoke({"question": request.inputs["question"], "answer": answer})
        print(new_answer)

        return answer

    fn, candidates, _ = optim.perform(
        langchain_chain,
        evaluator=ZenParea.metric_evaluator(data=evalset, n_workers=1, p=parea),
        samples=SAMPLES,
        rounds=1,
    )

    assert fn is not None
    assert any(candidates)
    assert next(e for e in candidates if 0.5 <= e.evals["score"] <= 1)


@pytest.mark.helpers
def test_zen_parea_helper_bootstrap_few_shot(
    parea: Parea, evalset: list, zen_parea_helper: ZenParea, openai, train_set, test_set, validation_set
):
    zenbase_manager = ZenbaseTracer()

    def score_answer_with_json(log: Log) -> EvaluationResult:
        output = str(expand_nested_json(log.output)["answer"])
        target = log.target.split("#### ")[-1]
        return EvaluationResult("correctness", int(output == target))

    @zenbase_manager.trace_function
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(max=8),
        before_sleep=before_sleep_log(log, logging.WARN),
    )
    @trace(eval_funcs=[score_answer_with_json])
    def solver(request: LMRequest):
        messages = [  # noqa
            {
                "role": "system",
                "content": "You are an expert math solver. You have a question that you should answer. You have step by step actions that you should take to solve the problem. You have the operations that you should do to solve the problem. You should come just with the number for the answer, just the actual number like examples that you have. Follow the format of the examples as they have the final answer, you need to come up with the plan for solving them."  # noqa
                'return it with json like return it in the {"answer": " the answer "}',
            }
        ]

        for demo in request.zenbase.task_demos:
            messages += [
                {"role": "user", "content": f"Example Question: {str(demo.inputs)}"},
                {"role": "assistant", "content": f"Example Answer: {str(demo.outputs)}"},
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
    @trace
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
                    {"role": "user", "content": str(demo.inputs)},
                    {"role": "assistant", "content": str(demo.outputs)},
                ]
        if not request.inputs:
            messages.append({"role": "user", "content": "Question: What is 2 + 2?"})
        else:
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
    @trace
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
                    {"role": "user", "content": f"Input: {str(demo.inputs)}"},
                    {"role": "assistant", "content": str(demo.outputs)},
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

    evaluator_kwargs = dict(p=parea, n_workers=1)
    bootstrap_few_shot = BootstrapFewShot(
        shots=SHOTS,
        training_set=train_set,
        test_set=test_set,
        validation_set=validation_set,
        evaluator_kwargs=evaluator_kwargs,
        zen_adaptor=zen_parea_helper,
    )

    best_lm, candidates = bootstrap_few_shot.perform(
        solver,
        samples=SAMPLES,
        rounds=1,
        trace_manager=zenbase_manager,
    )

    assert best_lm is not None

    zenbase_manager.all_traces = {}
    best_lm({"question": "What is 2 + 2?"})

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
def test_zen_parea_helper_bootstrap_few_shot_load_args(
    parea: Parea, evalset: list, zen_parea_helper: ZenParea, openai, train_set, test_set, validation_set
):
    zenbase_manager = ZenbaseTracer()

    def score_answer_with_json(log: Log) -> EvaluationResult:
        output = str(expand_nested_json(log.output)["answer"])
        target = log.target.split("#### ")[-1]
        return EvaluationResult("correctness", int(output == target))

    @zenbase_manager.trace_function
    @trace(eval_funcs=[score_answer_with_json])
    def solver(request: LMRequest):
        messages = [  # noqa
            {
                "role": "system",
                "content": "You are an expert math solver. You have a question that you should answer. You have step by step actions that you should take to solve the problem. You have the operations that you should do to solve the problem. You should come just with the number for the answer, just the actual number like examples that you have. Follow the format of the examples as they have the final answer, you need to come up with the plan for solving them."  # noqa
                'return it with json like return it in the {"answer": " the answer "}',
            }
        ]

        for demo in request.zenbase.task_demos:
            messages += [
                {"role": "user", "content": f"Example Question: {str(demo.inputs)}"},
                {"role": "assistant", "content": f"Example Answer: {str(demo.outputs)}"},
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
    @trace
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
                    {"role": "user", "content": str(demo.inputs)},
                    {"role": "assistant", "content": str(demo.outputs)},
                ]
        if not request.inputs:
            messages.append({"role": "user", "content": "Question: What is 2 + 2?"})
        else:
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
    @trace
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
                    {"role": "user", "content": f"Input: {str(demo.inputs)}"},
                    {"role": "assistant", "content": str(demo.outputs)},
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

    path_of_the_file = os.path.join(TEST_DIR, "adaptors/parea_bootstrap_few_shot.zenbase")

    best_lm = BootstrapFewShot.load_optimizer_and_function(
        optimizer_args_file=path_of_the_file, student_lm=solver, trace_manager=zenbase_manager
    )
    assert best_lm is not None

    zenbase_manager.all_traces = {}
    best_lm({"question": "What is 2 + 2?"})

    assert [v for k, v in zenbase_manager.all_traces.items()][0]["optimized"]["planner_chain"]["args"][
        "request"
    ].zenbase.task_demos[0].inputs is not None
    assert [v for k, v in zenbase_manager.all_traces.items()][0]["optimized"]["operation_finder"]["args"][
        "request"
    ].zenbase.task_demos[0].inputs is not None
    assert [v for k, v in zenbase_manager.all_traces.items()][0]["optimized"]["solver"]["args"][
        "request"
    ].zenbase.task_demos[0].inputs is not None
