import logging

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

        chain = ChatPromptTemplate.from_messages(messages) | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

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
