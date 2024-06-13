import logging

from openai import OpenAI
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
)
import pytest
import lunary

from zenbase.helpers.lunary import ZenLunary
from zenbase.optim.metric.labeled_few_shot import LabeledFewShot
from zenbase.types import LMRequest, deflm


SAMPLES = 2
SHOTS = 3
TESTSET_SIZE = 5

log = logging.getLogger(__name__)


@pytest.fixture
def optim(gsm8k_demoset: list):
    return LabeledFewShot(demoset=gsm8k_demoset, shots=SHOTS)


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


@deflm
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(max=8),
    before_sleep=before_sleep_log(log, logging.WARN),
)
def langchain_chain(request: LMRequest):
    """
    A math solver llm call that can solve any math problem setup with langchain libra.
    """

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    messages = [
        (
            "system",
            "You are an expert math solver. Your answer must be just the number with no separators, and nothing else. Follow the format of the examples.",
        )
    ]
    for demo in request.zenbase.demos:
        messages += [
            ("user", demo.inputs["question"]),
            ("assistant", demo.outputs["answer"]),
        ]

    messages.append(("user", "{question}"))

    chain = (
        ChatPromptTemplate.from_messages(messages)
        | ChatOpenAI(model="gpt-3.5-turbo")
        | StrOutputParser()
    )

    print("Mathing...")
    answer = chain.invoke(request.inputs)
    return answer.split("#### ")[-1]


@pytest.mark.helpers
def test_lunary_lcel_labeled_few_shot(optim: LabeledFewShot, evalset: list):
    fn, candidates = optim.perform(
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
