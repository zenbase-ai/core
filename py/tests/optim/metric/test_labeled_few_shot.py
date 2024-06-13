from random import Random, random
import pytest

from zenbase.optim.metric.labeled_few_shot import LabeledFewShot
from zenbase.optim.metric.types import CandidateMetricResult
from zenbase.types import LMDemo, LMFunction, LMRequest, deflm


lmfn = deflm(lambda x: x)


demoset = [
    LMDemo(inputs={}, outputs={"output": "a"}),
    LMDemo(inputs={}, outputs={"output": "b"}),
    LMDemo(inputs={}, outputs={"output": "c"}),
    LMDemo(inputs={}, outputs={"output": "d"}),
    LMDemo(inputs={}, outputs={"output": "e"}),
    LMDemo(inputs={}, outputs={"output": "f"}),
]


def test_invalid_shots():
    with pytest.raises(AssertionError):
        LabeledFewShot(demoset=demoset, shots=0)
    with pytest.raises(AssertionError):
        LabeledFewShot(demoset=demoset, shots=len(demoset) + 1)


def test_idempotency():
    shots = 2
    samples = 5

    optim1 = LabeledFewShot(demoset=demoset, shots=shots)
    optim2 = LabeledFewShot(demoset=demoset, shots=shots)
    optim3 = LabeledFewShot(demoset=demoset, shots=shots, random=Random(41))

    set1 = list(optim1.candidates(lmfn, samples))
    set2 = list(optim2.candidates(lmfn, samples))
    set3 = list(optim3.candidates(lmfn, samples))

    assert set1 == set2
    assert set1 != set3
    assert set2 != set3


@pytest.fixture
def optim():
    return LabeledFewShot(demoset=demoset, shots=2)


def test_candidate_generation(optim: LabeledFewShot):
    samples = 5

    candidates = list(optim.candidates(lmfn, samples))

    assert all(len(c.demos) == optim.shots for c in candidates)
    assert len(candidates) == samples


@deflm
def dummy_lmfn(_: LMRequest):
    return {"answer": 42}


def dummy_evalfn(fn: LMFunction):
    return CandidateMetricResult(fn, {"score": random()})


def test_training(optim: LabeledFewShot):
    # Train the dummy function
    trained_lmfn, candidates = optim.perform(
        dummy_lmfn,
        dummy_evalfn,
        rounds=1,
        concurrency=1,
    )

    # Check that the best function is returned
    best_function = max(candidates, key=lambda r: r.evals["score"]).function
    assert trained_lmfn == best_function

    for demo in trained_lmfn.zenbase.demos:
        assert demo in demoset


@pytest.mark.anyio
async def test_async_training(optim: LabeledFewShot):
    # Train the dummy function
    trained_dummy_lmfn, candidates = await optim.aperform(
        dummy_lmfn,
        dummy_evalfn,
        rounds=1,
        concurrency=1,
    )

    # Check that the best function is returned
    best_function = max(candidates, key=lambda r: r.evals["score"]).function
    assert trained_dummy_lmfn == best_function
