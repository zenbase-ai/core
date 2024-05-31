import pytest


@pytest.fixture(scope="session", autouse=True)
def env():
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(str(Path(__file__).parent.parent / ".env.test"))


@pytest.fixture
def gsm8k_dataset():
    import datasets

    return datasets.load_dataset("gsm8k", "main")


@pytest.fixture(scope="session", autouse=True)
def vcr_config():
    return {
        "filter_headers": ["authorization", "x-api-key"],
        "filter_query_parameters": ["api_key"],
        "cassette_library_dir": "tests/cache/cassettes",
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
        "record_mode": "new_episodes",
    }
