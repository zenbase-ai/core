import instructor
import pytest
from instructor.client import Instructor
from openai import OpenAI
from pydantic import BaseModel

from zenbase.core.managers import ZenbaseTracer
from zenbase.predefined.single_class_classifier.function_generator import SingleClassClassifierLMFunctionGenerator


@pytest.fixture(scope="module")
def prompt_definition() -> str:
    return """Your task is to accurately categorize each incoming email into one of the categories"""


@pytest.fixture(scope="module")
def class_dict() -> dict[str, str]:
    return {
        "spam": "Unsolicited and irrelevant emails, often advertising products or services, including phishing "
        "attempts and scams.",
        "promotional": "Emails related to sales, discounts, or marketing campaigns from businesses or services.",
        "social": "Emails originating from social media platforms, including notifications, friend requests, "
        "and updates.",
        "work": "Emails related to professional or business matters, including communications from colleagues, "
        "clients, and employers.",
        "personal": "Emails from friends, family, or acquaintances that are personal in nature, unrelated to work or "
        "promotions.",
    }


@pytest.fixture(scope="module")
def sample_promotional_email() -> dict[str, str]:
    return {
        "subject": "Limited Time Offer: 50% Off All Items!",
        "sender": "marketing@shoppingdeals.com",
        "body": """
    Hi there!

    We're excited to announce a limited time offer just for you. Enjoy 50% off all items in our store!
    Hurry, this offer is only valid until the end of the week.

    Best regards,
    The Shopping Deals Team
    """,
    }


@pytest.fixture(scope="module")
def openai_client() -> OpenAI:
    return OpenAI()


@pytest.fixture(scope="module")
def instructor_client(openai_client: OpenAI) -> Instructor:
    return instructor.from_openai(openai_client)


@pytest.fixture(scope="module")
def zenbase_tracer() -> ZenbaseTracer:
    return ZenbaseTracer()


@pytest.fixture(scope="module")
def single_class_classifier(
    instructor_client: Instructor, prompt_definition: str, class_dict: dict[str, str], zenbase_tracer: ZenbaseTracer
) -> SingleClassClassifierLMFunctionGenerator:
    return SingleClassClassifierLMFunctionGenerator(
        instructor_client=instructor_client,
        prompt=prompt_definition,
        class_dict=class_dict,
        model="gpt-3.5-turbo",
        zenbase_tracer=zenbase_tracer,
    )


def test_single_class_classifier_initialization(single_class_classifier: SingleClassClassifierLMFunctionGenerator):
    assert single_class_classifier is not None
    assert single_class_classifier.instructor_client is not None
    assert single_class_classifier.prompt is not None
    assert single_class_classifier.class_dict is not None
    assert single_class_classifier.model is not None
    assert single_class_classifier.zenbase_tracer is not None

    # Check generated class enum and prediction class
    assert single_class_classifier.class_enum is not None
    assert issubclass(single_class_classifier.prediction_class, BaseModel)


def test_single_class_classifier_prediction(
    single_class_classifier: SingleClassClassifierLMFunctionGenerator, sample_promotional_email
):
    result = single_class_classifier.generate()(sample_promotional_email)

    assert result.class_label.name == "promotional"
    assert single_class_classifier.zenbase_tracer.all_traces is not None


def test_single_class_classifier_with_missing_data(single_class_classifier: SingleClassClassifierLMFunctionGenerator):
    faulty_email = {"subject": "Meeting Reminder"}
    result = single_class_classifier.generate()(faulty_email)

    # Test how the classifier handles missing data
    assert result.class_label.name in {"work", "personal", "social", "spam", "promotional"}
