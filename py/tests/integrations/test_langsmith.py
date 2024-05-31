import json

from datasets import DatasetDict
from langsmith import Client, traceable
from langsmith.schemas import Run, Example
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI
import pytest
import requests


from zenbase.optimizers.labelled_few_shot import LabelledFewShot
from zenbase.integrations.langsmith import LangSmithZen
from zenbase.types import LMPrompt


@pytest.fixture
def langsmith():
    return Client()


@pytest.fixture
def openai():
    return wrap_openai(AsyncOpenAI())


TEST_SIZE = 5
SAMPLE_SIZE = 2


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


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_langsmith_zen_labelled_few_shot(
    langsmith: Client,
    openai: AsyncOpenAI,
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

        few_shot_examples = prompt["examples"]

        messages = [
            {
                "role": "system",
                "content": "You are an expert math solver. Your answer must be just the number with no separators, and nothing else. Follow the format of the examples. Respond with a JSON object.",
            },
        ]
        for example in few_shot_examples:
            messages.append({"role": "user", "content": json.dumps(example["inputs"])})
            messages.append(
                {"role": "assistant", "content": json.dumps(example["outputs"])}
            )
        messages.append({"role": "user", "content": json.dumps({"question": question})})

        print("Mathing...")
        response = await openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)

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

    optimized_function, run = await LabelledFewShot.optimize(
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
