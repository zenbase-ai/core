from anyio._core._eventloop import threadlocals
from typing import AsyncIterable, Awaitable, Callable, ParamSpec, TypeVar
from random import Random
from concurrent.futures import ThreadPoolExecutor
import anyio
import asyncio
import functools
import inspect
import logging
import os

from faker import Faker
from opentelemetry import trace
from pksuid import PKSUID
from structlog import get_logger


get_logger: Callable[..., logging.Logger] = get_logger
tracer = trace.get_tracer("zenbase")


def get_seed(seed: int | None = None) -> int:
    return seed or int(os.getenv("RANDOM_SEED", 42))


def random_factory(seed: int | None = None) -> Random:
    return Random(get_seed(seed))


def id_generator(prefix: str) -> Callable[[], str]:
    return lambda: str(PKSUID(prefix))


def random_name_generator(
    prefix: str | None = None,
    random_name_generator=Faker().catch_phrase,
) -> Callable[[], str]:
    head = f"zenbase-{prefix}" if prefix else "zenbase"

    def gen():
        tail = random_name_generator().lower().replace(" ", "-")
        return f"{head}-{tail}"

    return gen


I_ParamSpec = ParamSpec("I_ParamSpec")
O_Retval = TypeVar("O_Retval")


def asyncify(
    func: Callable[I_ParamSpec, O_Retval],
    *,
    cancellable: bool = True,
    limiter: anyio.CapacityLimiter | None = None,
) -> Callable[I_ParamSpec, Awaitable[O_Retval]]:
    if inspect.iscoroutinefunction(func):
        return func

    @functools.wraps(func)
    async def wrapper(
        *args: I_ParamSpec.args, **kwargs: I_ParamSpec.kwargs
    ) -> O_Retval:
        partial_f = functools.partial(func, *args, **kwargs)
        return await anyio.to_thread.run_sync(
            partial_f,
            abandon_on_cancel=cancellable,
            limiter=limiter,
        )

    return wrapper


def syncify(
    func: Callable[I_ParamSpec, O_Retval],
) -> Callable[I_ParamSpec, O_Retval]:
    if not inspect.iscoroutinefunction(func):
        return func

    @functools.wraps(func)
    def wrapper(*args: I_ParamSpec.args, **kwargs: I_ParamSpec.kwargs) -> O_Retval:
        partial_f = functools.partial(func, *args, **kwargs)
        if not getattr(threadlocals, "current_async_backend", None):
            try:
                return asyncio.get_running_loop().run_until_complete(partial_f())
            except RuntimeError:
                return anyio.run(partial_f)
        return anyio.from_thread.run(partial_f)

    return wrapper


async def amap[
    O
](func: Callable[..., Awaitable[O]], iterable, *iterables, concurrency=10) -> list[O]:
    assert concurrency >= 1, "Concurrency must be greater than or equal to 1"

    if concurrency == 1:
        return [await func(*args) for args in zip(iterable, *iterables)]

    if concurrency == float("inf"):
        return await asyncio.gather(
            *[func(*args) for args in zip(iterable, *iterables)]
        )

    semaphore = asyncio.Semaphore(concurrency)

    @functools.wraps(func)
    async def mapper(*args):
        async with semaphore:
            return await func(*args)

    return await asyncio.gather(*[mapper(*args) for args in zip(iterable, *iterables)])


def pmap[O](func: Callable[..., O], iterable, *iterables, concurrency=10) -> list[O]:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(func, iterable, *iterables))


async def alist[O](aiterable: AsyncIterable[O]) -> list[O]:
    return [x async for x in aiterable]
