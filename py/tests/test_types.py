import asyncio

import pytest

from zenbase.types import LMDemo, deflm, use_zenbase


def test_demo_eq():
    demoset = [
        LMDemo(inputs={}, outputs={"output": "a"}),
        LMDemo(inputs={}, outputs={"output": "b"}),
    ]

    # Structural inequality
    assert demoset[0] != demoset[1]
    # Structural equality
    assert demoset[0] == LMDemo(inputs={}, outputs={"output": "a"})


def test_lm_function_refine():
    fn = deflm(lambda r: r.inputs)
    assert fn != fn.refine()


@pytest.mark.anyio
async def test_lm_function_async():
    fn = deflm(lambda r: r.inputs)
    assert fn({"answer": 42}) == await fn.coro({"answer": 42})


@pytest.mark.anyio
async def test_async_lm_function_zenbase_context():
    @deflm
    async def l2_fn2(r):
        assert r.zenbase == use_zenbase()

    @deflm
    async def l2_fn1(r):
        assert r.zenbase == use_zenbase()

    @deflm
    async def l1_fn(r):
        await asyncio.gather(l2_fn1.coro(r.inputs), l2_fn2.coro(r.inputs))
        assert r.zenbase == use_zenbase()

    @deflm
    async def l0_fn(r):
        await l1_fn.coro(r.inputs)
        assert r.zenbase == use_zenbase()

    await l0_fn.coro({})


def test_lm_function_zenbase_context():
    @deflm
    def l2_fn2(r):
        assert r.zenbase == use_zenbase()

    @deflm
    def l2_fn1(r):
        assert r.zenbase == use_zenbase()

    @deflm
    def l1_fn(r):
        l2_fn1(r.inputs)
        l2_fn2(r.inputs)
        assert r.zenbase == use_zenbase()

    @deflm
    def l0_fn(r):
        l1_fn(r.inputs)
        assert r.zenbase == use_zenbase()

    l0_fn({})
