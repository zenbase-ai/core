import logging

from openai import OpenAI
from langfuse import Langfuse
from langfuse.decorators import observe
from datasets import DatasetDict
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
)
import pytest

from zenbase.helpers.langfuse import ZenLangfuse
from zenbase.optim.metric.labeled_few_shot import LabeledFewShot
from zenbase.optim.metric.types import MetricEvals
from zenbase.types import LMDemo, LMRequest, deflm
from zenbase.utils import pmap

SAMPLES = 2
SHOTS = 3
EVALSET_SIZE = 5

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


@deflm
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(max=8),
    before_sleep=before_sleep_log(log, logging.WARN),
)
@observe()
def langchain_chain(request: LMRequest) -> str:
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
    return answer


def score_answer(answer: str, demo: LMDemo, langfuse: Langfuse) -> MetricEvals:
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
    fn, candidates = optim.perform(
        langchain_chain,
        evaluator=ZenLangfuse.metric_evaluator(evalset, evaluate=score_answer),
        samples=SAMPLES,
        rounds=1,
    )

    assert fn is not None
    assert any(candidates)
    assert next(e for e in candidates if 0.5 <= e.evals["score"] <= 1)
