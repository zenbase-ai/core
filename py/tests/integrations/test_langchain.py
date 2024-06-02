import json

from datasets import DatasetDict
from langsmith import Client, traceable
from langsmith.schemas import Run, Example
from langsmith.wrappers import wrap_openai
import pytest
import requests


from zenbase.integrations.langchain import LangSmithZen
from zenbase.optimizers.labeled_few_shot import LabeledFewShot
from zenbase.types import LMPrompt

TEST_SIZE = 5
SAMPLE_SIZE = 2


@pytest.fixture
def langsmith():
    return Client()


@pytest.fixture
def golden_demos(gsm8k_dataset: DatasetDict):
    return [
        {"inputs": {"question": r["question"]}, "outputs": {"answer": r["answer"]}}
        for r in gsm8k_dataset["train"].select(range(5))
    ]


@pytest.fixture
@pytest.mark.vcr
def test_examples(gsm8k_dataset: DatasetDict, langsmith: Client):
    try:
        return list(langsmith.list_examples(dataset_name="gsm8k-test-examples"))
    except requests.exceptions.HTTPError as e:
        if e.response.status_code != 404:
            raise
        dataset = langsmith.create_dataset("gsm8k-test-examples")
        examples = gsm8k_dataset["test"].select(TEST_SIZE)
        langsmith.create_examples(
            inputs=[{"question": e["question"]} for e in examples],
            outputs=[{"answer": e["answer"]} for e in examples],
            dataset_id=dataset.id,
        )
        return list(langsmith.list_examples(dataset_name="gsm8k-test-examples"))


def score_answer(run: Run, example: Example) -> bool:
    return {
        "key": "correctness",
        "score": int(
            run.outputs["answer"].split("#### ")[-1]
            == example.outputs["answer"].split("#### ")[-1]
        ),
    }


def score_experiment(runs: list[Run], examples: list[Example]) -> dict:
    return {
        "key": "accuracy",
        "score": sum(score_answer(r, e)["score"] for r, e in zip(runs, examples))
        / len(examples),
    }


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_langchain_labelled_few_shot(
    langsmith: Client,
    test_examples: list,
    golden_demos: list,
):
    async def function(question: str, prompt: LMPrompt, return_prompt: bool = False):
        if return_prompt:
            return prompt

        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        few_shot_examples = prompt["examples"]

        prompt_messages = [
            (
                "system",
                "You are an expert math solver. Your answer must be just the number with no separators, and nothing else. Follow the format of the examples.",
            )
        ]
        for example in few_shot_examples:
            prompt_messages += [
                ("user", example["inputs"]["question"]),
                ("assistant", example["outputs"]["answer"]),
            ]

        prompt_messages.append(("user", question))

        chain = (
            ChatPromptTemplate.from_messages(prompt_messages)
            | ChatOpenAI(model="gpt-3.5-turbo")
            | StrOutputParser()
        )
        answer = await chain.ainvoke({"question": question})
        return {"answer": answer}

    optimized_function, run = await LabeledFewShot.optimize(
        function,
        samples=SAMPLE_SIZE,
        demos=golden_demos,
        evaluator=LangSmithZen.evaluator(
            test_examples,
            evaluators=[score_answer],
            client=langsmith,
        ),
    )

    prompt = await optimized_function(question="ignored", return_prompt=True)
    assert prompt == run["winner"]["prompt"]


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_langsmith_zen_labelled_few_shot(
    langsmith: Client,
    golden_demos: list,
    test_examples: list,
):
    @traceable
    async def function(
        question: str,
        prompt: LMPrompt,
        return_prompt: bool = False,
    ) -> dict:
        if return_prompt:
            return prompt

        from openai import AsyncOpenAI

        openai = wrap_openai(AsyncOpenAI())
        few_shot_examples = prompt["examples"]

        messages = [
            {
                "role": "system",
                "content": "You are an expert math solver. Your answer must be just the number with no separators, and nothing else. Follow the format of the examples. Respond with a JSON object.",
            },
        ]
        for example in few_shot_examples:
            messages += [
                {"role": "user", "content": json.dumps(example["inputs"])},
                {"role": "assistant", "content": json.dumps(example["outputs"])},
            ]
        messages.append({"role": "user", "content": json.dumps({"question": question})})

        print("Mathing...")
        response = await openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)

    optimized_function, run = await LabeledFewShot.optimize(
        function,
        samples=SAMPLE_SIZE,
        demos=golden_demos,
        evaluator=LangSmithZen.evaluator(
            test_examples,
            evaluators=[score_answer],
            summary_evaluators=[score_experiment],
            client=langsmith,
        ),
    )

    prompt = await optimized_function(question="ignored", return_prompt=True)
    assert prompt == run["winner"]["prompt"]
