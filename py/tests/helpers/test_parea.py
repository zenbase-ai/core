import json
import logging

from openai import OpenAI
from parea import Parea, trace
from parea.schemas import Log, EvaluationResult
from datasets import DatasetDict
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
)
import pytest

from zenbase.helpers.parea import ZenParea
from zenbase.optim.metric.labeled_few_shot import LabeledFewShot
from zenbase.types import LMRequest, deflm

SAMPLES = 2
SHOTS = 3
EVALSET_SIZE = 5

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


def score_answer(log: Log) -> EvaluationResult:
    output = log.output.split("#### ")[-1]
    target = log.target.split("#### ")[-1]
    return EvaluationResult("correctness", int(output == target))


@deflm
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(max=8),
    before_sleep=before_sleep_log(log, logging.WARN),
)
@trace(eval_funcs=[score_answer])
def langchain_chain(request: LMRequest):
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

    chain_2 = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the verifier",
                ),
                ("user", "the question is: {question} and the answer is {answer}, is it right? answer with yes or no."),
            ]
        )
        | ChatOpenAI(model="gpt-3.5-turbo")
        | StrOutputParser()
    )

    print("Mathing...")
    new_answer = chain_2.invoke({'question': request.inputs['question'], 'answer': answer})
    print(new_answer)

    return answer


@pytest.mark.helpers
def test_parea_lcel_labeled_few_shot(
    optim: LabeledFewShot,
    parea: Parea,
    evalset: list,
):
    fn, candidates = optim.perform(
        langchain_chain,
        evaluator=ZenParea.metric_evaluator(data=evalset, n_workers=2, p=parea),
        samples=SAMPLES,
        rounds=1,
    )

    assert fn is not None
    assert any(candidates)
    assert next(e for e in candidates if 0.5 <= e.evals["score"] <= 1)


