from random import random
import pytest

from zenbase.optimizers.labelled_few_shot import LabelledFewShot
from zenbase.types import LMPrompt, LMEvaluatorRun, LMFunctionDemo


class TestCandidates:
    examples = [
        LMFunctionDemo(inputs={}, output="a"),
        LMFunctionDemo(inputs={}, output="b"),
        LMFunctionDemo(inputs={}, output="c"),
    ]

    def test_seed_idempotency(self):
        run_1 = list(LabelledFewShot.candidates(self.examples, shots=2))
        run_2 = list(LabelledFewShot.candidates(self.examples, shots=2))
        run_3 = list(LabelledFewShot.candidates(self.examples, shots=2, seed=41))

        assert run_1 == run_2
        assert run_1 != run_3
        assert run_2 != run_3

    def test_insufficient_examples(self):
        with pytest.raises(AssertionError):
            list(LabelledFewShot.candidates(self.examples, shots=5))

    def test_example_count(self):
        candidates = list(LabelledFewShot.candidates(self.examples, shots=2))

        assert all(len(candidate["examples"]) == 2 for candidate in candidates)
        assert len(candidates) == 6


class TestOptimizer:
    examples = [
        LMFunctionDemo(inputs={}, output="a"),
        LMFunctionDemo(inputs={}, output="b"),
        LMFunctionDemo(inputs={}, output="c"),
        LMFunctionDemo(inputs={}, output="d"),
        LMFunctionDemo(inputs={}, output="e"),
        LMFunctionDemo(inputs={}, output="f"),
    ]

    @pytest.mark.asyncio
    async def test_optimizer(self):
        async def dummy_predictor(*, prompt: LMPrompt) -> LMPrompt:
            return prompt

        async def dummy_evaluator(_, prompt: LMPrompt) -> LMEvaluatorRun:
            return {
                "prompt": prompt,
                "evals": {"score": random()},
                "function_runs": [],
            }

        optimized_predictor, run = await LabelledFewShot.optimize(
            fn=dummy_predictor,
            evaluator=dummy_evaluator,
            demos=self.examples,
            shots=2,
            samples=5,
        )

        best_run = max(run["candidates"], key=lambda r: r["evals"]["score"])
        assert best_run == run["winner"]

        winning_prompt = await optimized_predictor()
        assert winning_prompt == run["winner"]["prompt"]
