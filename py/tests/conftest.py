from typing import Any, Callable
from datasets import DatasetDict
import pytest

from zenbase.types import LMDemo


def pytest_configure():
    from dotenv import load_dotenv
    from pathlib import Path

    load_dotenv(str(Path(__file__).parent.parent / ".env.test"))

    import nest_asyncio

    nest_asyncio.apply()


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--helpers", action="store_true", help="run helpers tests")


def pytest_runtest_setup(item: pytest.Item):
    if "helpers" in item.keywords and not item.config.getoption("--helpers"):
        pytest.skip("skipping integration tests")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def vcr_config():
    def response_processor(exclude_headers: list[str]) -> Callable:
        def before_record_response(response: dict | Any) -> dict | Any:
            if isinstance(response, dict):
                try:
                    response_str = (
                        response.get("body", {}).get("string", b"").decode("utf-8")
                    )
                    if "Rate limit reached for" in response_str:
                        # don't record rate-limiting responses
                        return None
                except UnicodeDecodeError:
                    pass  # ignore if we can't parse response

                for header in exclude_headers:
                    if header in response["headers"]:
                        response["headers"].pop(header)
            return response

        return before_record_response

    return {
        "filter_headers": [
            "User-Agent",
            "Accept",
            "Accept-Encoding",
            "Connection",
            "Content-Length",
            "Content-Type",
            # OpenAI request headers we don't want
            "Cookie",
            "authorization",
            "X-OpenAI-Client-User-Agent",
            "OpenAI-Organization",
            "x-stainless-lang",
            "x-stainless-package-version",
            "x-stainless-os",
            "x-stainless-arch",
            "x-stainless-runtime",
            "x-stainless-runtime-version",
            "x-api-key",
        ],
        "filter_query_parameters": ["api_key"],
        "cassette_library_dir": "tests/cache/cassettes",
        "before_record_response": response_processor(
            exclude_headers=[
                # OpenAI response headers we don't want
                "Set-Cookie",
                "Server",
                "access-control-allow-origin",
                "alt-svc",
                "openai-organization",
                "openai-version",
                "strict-transport-security",
                "x-ratelimit-limit-requests",
                "x-ratelimit-limit-tokens",
                "x-ratelimit-remaining-requests",
                "x-ratelimit-remaining-tokens",
                "x-ratelimit-reset-requests",
                "x-ratelimit-reset-tokens",
                "x-request-id",
            ]
        ),
        "match_on": [
            "method",
            "scheme",
            "host",
            "port",
            "path",
            "query",
            "body",
            "headers",
        ],
    }


@pytest.fixture(scope="session")
def gsm8k_dataset():
    import datasets

    return datasets.load_dataset("gsm8k", "main")


@pytest.fixture(scope="session")
def gsm8k_demoset(gsm8k_dataset: DatasetDict) -> list[LMDemo]:
    return [
        LMDemo(
            inputs={"question": r["question"]},
            outputs={"answer": r["answer"]},
        )
        for r in gsm8k_dataset["train"].select(range(5))
    ]
