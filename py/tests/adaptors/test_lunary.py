import json
import logging
import os

import lunary
import pytest
from openai import OpenAI
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
)

from zenbase.adaptors.lunary import ZenLunary
from zenbase.core.managers import ZenbaseTracer
from zenbase.optim.metric.bootstrap_few_shot import BootstrapFewShot
from zenbase.optim.metric.labeled_few_shot import LabeledFewShot
from zenbase.settings import TEST_DIR
from zenbase.types import LMRequest

SAMPLES = 2
SHOTS = 3
TESTSET_SIZE = 5

log = logging.getLogger(__name__)


@pytest.fixture
def optim(gsm8k_demoset: list):
    return LabeledFewShot(demoset=gsm8k_demoset, shots=SHOTS)


@pytest.fixture
def bootstrap_few_shot_optim(gsm8k_demoset: list):
    return BootstrapFewShot(training_set_demos=gsm8k_demoset, shots=SHOTS)


@pytest.fixture(scope="module")
def openai():
    client = OpenAI()
    lunary.monitor(client)
    return client


@pytest.fixture(scope="module")
def evalset():
    items = lunary.get_dataset("gsm8k-evalset")
    assert any(items)
    return items


@pytest.mark.helpers
def test_lunary_lcel_labeled_few_shot(optim: LabeledFewShot, evalset: list):
    trace_manager = ZenbaseTracer()

    @trace_manager.trace_function
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(max=8),
        before_sleep=before_sleep_log(log, logging.WARN),
    )
    def langchain_chain(request: LMRequest):
        """
        A math solver llm call that can solve any math problem setup with langchain libra.
        """

        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        messages = [
            (
                "system",
                "You are an expert math solver. Your answer must be just the number with no separators, and nothing else. Follow the format of the examples.",  # noqa
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
        return answer.split("#### ")[-1]

    fn, candidates, _ = optim.perform(
        langchain_chain,
        evaluator=ZenLunary.metric_evaluator(
            checklist="exact-match",
            evalset=evalset,
            concurrency=2,
        ),
        samples=SAMPLES,
        rounds=1,
    )

    assert fn is not None
    assert any(candidates)
    assert next(c for c in candidates if 0.5 <= c.evals["score"] <= 1)


@pytest.fixture(scope="module")
def lunary_helper():
    return ZenLunary(client=lunary)


@pytest.mark.helpers
def test_lunary_openai_bootstrap_few_shot(optim: LabeledFewShot, lunary_helper: ZenLunary, openai):
    zenbase_manager = ZenbaseTracer()

    @zenbase_manager.trace_function
    # @retry(
    #     stop=stop_after_attempt(3),
    #     wait=wait_exponential_jitter(max=8),
    #     before_sleep=before_sleep_log(log, logging.WARN),
    # )
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
                {"role": "user", "content": f"Example Question: {demo.inputs}"},
                {"role": "assistant", "content": f"Example Answer: {demo.outputs}"},
            ]

        plan = planner_chain(request.inputs)
        the_plan = plan["plan"]
        # the_plan = 'plan["plan"]'
        the_operation = operation_finder(
            {
                "plan": the_plan,
                "question": request.inputs,
            }
        )
        # the_operation = {"operation": "operation_finder"}

        messages.append({"role": "user", "content": f"Question: {request.inputs}"})
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
        return answer["answer"]

    @zenbase_manager.trace_function
    # @retry(
    #     stop=stop_after_attempt(3),
    #     wait=wait_exponential_jitter(max=8),
    #     before_sleep=before_sleep_log(log, logging.WARN),
    # )
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
                    {"role": "user", "content": demo.inputs},
                    {"role": "assistant", "content": demo.outputs["plan"]},
                ]
        messages.append({"role": "user", "content": request.inputs})

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            response_format={"type": "json_object"},
        )

        print("Planning...")
        answer = json.loads(response.choices[0].message.content)
        return {"plan": " ".join(i for i in answer["plan"])}

    @zenbase_manager.trace_function
    # @retry(
    #     stop=stop_after_attempt(3),
    #     wait=wait_exponential_jitter(max=8),
    #     before_sleep=before_sleep_log(log, logging.WARN),
    # )
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

    solver("What is 2 + 2?")

    evaluator_kwargs = dict(
        checklist="exact-match",
        concurrency=2,
    )

    # for lunary there is not feature to create dataset with code, so dataset are created
    # manually with UI, if you want to replicate the test on your own, you should put
    # GSM8K examples to dataset name like below:
    TRAIN_SET = "gsmk8k-train-set"
    TEST_SET = "gsm8k-test-set"
    VALIDATION_SET = "gsm8k-validation-set"

    assert lunary_helper.fetch_dataset_demos(TRAIN_SET) is not None
    assert lunary_helper.fetch_dataset_demos(TEST_SET) is not None
    assert lunary_helper.fetch_dataset_demos(VALIDATION_SET) is not None

    bootstrap_few_shot = BootstrapFewShot(
        shots=SHOTS,
        training_set=TRAIN_SET,
        test_set=TEST_SET,
        validation_set=VALIDATION_SET,
        evaluator_kwargs=evaluator_kwargs,
        zen_adaptor=lunary_helper,
    )

    teacher_lm, candidates = bootstrap_few_shot.perform(
        solver,
        samples=SAMPLES,
        rounds=1,
        trace_manager=zenbase_manager,
    )
    assert teacher_lm is not None

    zenbase_manager.all_traces = {}
    teacher_lm("What is 2 + 2?")

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
