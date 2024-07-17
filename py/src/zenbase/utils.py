import asyncio
import functools
import inspect
import json
import logging
import os
from random import Random
from typing import AsyncIterable, Awaitable, Callable, ParamSpec, TypeVar

import anyio
from anyio._core._eventloop import threadlocals
from faker import Faker
from opentelemetry import trace
from pksuid import PKSUID
from posthog import Posthog
from structlog import get_logger

get_logger: Callable[..., logging.Logger] = get_logger
ot_tracer = trace.get_tracer("zenbase")


def posthog() -> Posthog:
    if project_api_key := os.getenv("ZENBASE_ANALYTICS_KEY"):
        client = Posthog(
            project_api_key=project_api_key,
            host="https://us.i.posthog.com",
        )
        client.identify(os.environ["ZENBASE_ANALYTICS_ID"])
    else:
        client = Posthog("")
        client.disabled = True
    return client


def get_seed(seed: int | None = None) -> int:
    return seed or int(os.getenv("RANDOM_SEED", 42))


def random_factory(seed: int | None = None) -> Random:
    return Random(get_seed(seed))


def ksuid(prefix: str | None = None) -> str:
    return str(PKSUID(prefix))


def ksuid_generator(prefix: str) -> Callable[[], str]:
    return functools.partial(ksuid, prefix)


def random_name_generator(
    prefix: str | None = None,
    random_name_generator=Faker().catch_phrase,
) -> Callable[[], str]:
    head = f"zenbase-{prefix}" if prefix else "zenbase"

    def gen():
        return "-".join([head, *random_name_generator().lower().split(" ")[:2]])

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
    async def wrapper(*args: I_ParamSpec.args, **kwargs: I_ParamSpec.kwargs) -> O_Retval:
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


ReturnValue = TypeVar("ReturnValue", covariant=True)


async def amap(
    func: Callable[..., Awaitable[ReturnValue]],
    iterable,
    *iterables,
    concurrency=10,
) -> list[ReturnValue]:
    assert concurrency >= 1, "Concurrency must be greater than or equal to 1"

    if concurrency == 1:
        return [await func(*args) for args in zip(iterable, *iterables)]

    if concurrency == float("inf"):
        return await asyncio.gather(*[func(*args) for args in zip(iterable, *iterables)])

    semaphore = asyncio.Semaphore(concurrency)

    @functools.wraps(func)
    async def mapper(*args):
        async with semaphore:
            return await func(*args)

    return await asyncio.gather(*[mapper(*args) for args in zip(iterable, *iterables)])


def pmap(
    func: Callable[..., ReturnValue],
    iterable,
    *iterables,
    concurrency=10,
) -> list[ReturnValue]:
    # TODO: Should revert.
    return [func(*args) for args in zip(iterable, *iterables)]


async def alist(aiterable: AsyncIterable[ReturnValue]) -> list[ReturnValue]:
    return [x async for x in aiterable]


def expand_nested_json(d):
    def recursive_expand(value):
        if isinstance(value, str):
            try:
                # Try to parse the string value as JSON
                parsed_value = json.loads(value)
                # Recursively expand the parsed value in case it contains further nested JSON
                return recursive_expand(parsed_value)
            except json.JSONDecodeError:
                # If parsing fails, return the original string
                return value
        elif isinstance(value, dict):
            # Recursively expand each key-value pair in the dictionary
            return {k: recursive_expand(v) for k, v in value.items()}
        elif isinstance(value, list):
            # Recursively expand each element in the list
            return [recursive_expand(elem) for elem in value]
        else:
            return value

    return recursive_expand(d)
