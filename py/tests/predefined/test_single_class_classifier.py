import datasets
import instructor
import pandas as pd
import pytest
from instructor.client import Instructor
from openai import OpenAI
from pydantic import BaseModel

from zenbase.core.managers import ZenbaseTracer
from zenbase.predefined.single_class_classifier import SingleClassClassifier
from zenbase.predefined.single_class_classifier.function_generator import SingleClassClassifierLMFunctionGenerator

TRAINSET_SIZE = 100
VALIDATIONSET_SIZE = 21
TESTSET_SIZE = 21


@pytest.fixture(scope="module")
def prompt_definition() -> str:
    return """Your task is to accurately categorize each incoming news article into one of the given categories based
    on its title and content."""


@pytest.fixture(scope="module")
def class_dict() -> dict[str, str]:
    return {
        "Automobiles": "Discussions and news about automobiles, including car maintenance, driving experiences, "
        "and the latest automotive technology.",
        "Computers": "Topics related to computer hardware, software, graphics, cryptography, and operating systems, "
        "including troubleshooting and advancements.",
        "Science": "News and discussions about scientific topics including space exploration, medicine, and "
        "electronics.",
        "Politics": "Debates and news about political topics, including gun control, Middle Eastern politics,"
        " and miscellaneous political discussions.",
        "Religion": "Discussions about various religions, including beliefs, practices, atheism, and religious news.",
        "For Sale": "Classified ads for buying and selling miscellaneous items, from electronics to household goods.",
        "Sports": "Everything about sports, including discussions, news, player updates, and game analysis.",
    }


@pytest.fixture(scope="module")
def sample_news_article():
    return """title: New Advancements in Electric Vehicle Technology
                content: The automotive industry is witnessing a significant shift towards electric vehicles (EVs).
                Recent advancements in battery technology have led to increased range and reduced charging times.
                Companies like Tesla, Nissan, and BMW are at the forefront of this innovation, aiming to make EVs
                more accessible and efficient.
                With governments worldwide pushing for greener alternatives, the future of transportation looks
                electric."""


def create_dataset_with_examples(item_set: list):
    return [{"inputs": item["text"], "outputs": convert_to_human_readable(item["label_text"])} for item in item_set]


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
def single_class_classifier_generator(
    instructor_client: Instructor, prompt_definition: str, class_dict: dict[str, str], zenbase_tracer: ZenbaseTracer
) -> SingleClassClassifierLMFunctionGenerator:
    return SingleClassClassifierLMFunctionGenerator(
        instructor_client=instructor_client,
        prompt=prompt_definition,
        class_dict=class_dict,
        model="gpt-4o-mini",
        zenbase_tracer=zenbase_tracer,
    )


@pytest.mark.helpers
def test_single_class_classifier_lm_function_generator_initialization(
    single_class_classifier_generator: SingleClassClassifierLMFunctionGenerator,
):
    assert single_class_classifier_generator is not None
    assert single_class_classifier_generator.instructor_client
    None
    assert single_class_classifier_generator.class_dict is not None
    assert single_class_classifier_generator.model is not None
    assert single_class_classifier_generator.zenbase_tracer is not None

    # Check generated class enum and prediction class
    assert single_class_classifier_generator.class_enum is not None
    assert issubclass(single_class_classifier_generator.prediction_class, BaseModel)


@pytest.mark.helpers
def test_single_class_classifier_lm_function_generator_prediction(
    single_class_classifier_generator: SingleClassClassifierLMFunctionGenerator, sample_news_article
):
    result = single_class_classifier_generator.generate()(sample_news_article)

    assert result.class_label.name == "Automobiles"
    assert single_class_classifier_generator.zenbase_tracer.all_traces is not None


@pytest.mark.helpers
def test_single_class_classifier_lm_function_generator_with_missing_data(
    single_class_classifier_generator: SingleClassClassifierLMFunctionGenerator,
):
    faulty_email = {"subject": "Meeting Reminder"}
    result = single_class_classifier_generator.generate()(faulty_email)
    assert result.class_label.name in {
        "Automobiles",
        "Computers",
        "Science",
        "Politics",
        "Religion",
        "For Sale",
        "Sports",
        "Other",
    }


def convert_to_human_readable(category: str) -> str:
    human_readable_map = {
        "rec.autos": "Automobiles",
        "comp.sys.mac.hardware": "Computers",
        "comp.graphics": "Computers",
        "sci.space": "Science",
        "talk.politics.guns": "Politics",
        "sci.med": "Science",
        "comp.sys.ibm.pc.hardware": "Computers",
        "comp.os.ms-windows.misc": "Computers",
        "rec.motorcycles": "Automobiles",
        "talk.religion.misc": "Religion",
        "misc.forsale": "For Sale",
        "alt.atheism": "Religion",
        "sci.electronics": "Computers",
        "comp.windows.x": "Computers",
        "rec.sport.hockey": "Sports",
        "rec.sport.baseball": "Sports",
        "soc.religion.christian": "Religion",
        "talk.politics.mideast": "Politics",
        "talk.politics.misc": "Politics",
        "sci.crypt": "Computers",
    }
    return human_readable_map.get(category)


@pytest.fixture(scope="module")
def get_balanced_dataset():
    # Load the dataset
    split = "train"
    train_size = TRAINSET_SIZE
    validation_size = VALIDATIONSET_SIZE
    test_size = TESTSET_SIZE
    dataset = datasets.load_dataset("SetFit/20_newsgroups", split=split)

    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(dataset)

    # Convert text labels to human-readable labels
    df["human_readable_label"] = df["label_text"].apply(convert_to_human_readable)

    # Find text labels
    text_labels = df["human_readable_label"].unique()

    # Determine the number of labels
    num_labels = len(text_labels)

    # Calculate the number of samples per label for each subset
    train_samples_per_label = train_size // num_labels
    validation_samples_per_label = validation_size // num_labels
    test_samples_per_label = test_size // num_labels

    # Create empty DataFrames for train, validation, and test sets
    train_set = pd.DataFrame()
    validation_set = pd.DataFrame()
    test_set = pd.DataFrame()

    # Split sequentially without shuffling
    for label in text_labels:
        label_df = df[df["human_readable_label"] == label]

        # Ensure there's enough data
        if len(label_df) < (train_samples_per_label + validation_samples_per_label + test_samples_per_label):
            raise ValueError(f"Not enough data for label {label}")

        # Split the label-specific DataFrame
        label_train = label_df.iloc[:train_samples_per_label]
        label_validation = label_df.iloc[
            train_samples_per_label : train_samples_per_label + validation_samples_per_label
        ]
        label_test = label_df.iloc[
            train_samples_per_label + validation_samples_per_label : train_samples_per_label
            + validation_samples_per_label
            + test_samples_per_label
        ]

        # Append to the respective sets
        train_set = pd.concat([train_set, label_train])
        validation_set = pd.concat([validation_set, label_validation])
        test_set = pd.concat([test_set, label_test])

    # Create dataset with examples
    train_set = create_dataset_with_examples(train_set.to_dict("records"))
    validation_set = create_dataset_with_examples(validation_set.to_dict("records"))
    test_set = create_dataset_with_examples(test_set.to_dict("records"))

    return train_set, validation_set, test_set


@pytest.fixture(scope="module")
def single_class_classifier(
    instructor_client: Instructor,
    prompt_definition: str,
    class_dict: dict[str, str],
    zenbase_tracer: ZenbaseTracer,
    get_balanced_dataset,
) -> SingleClassClassifier:
    train_set, validation_set, test_set = get_balanced_dataset
    return SingleClassClassifier(
        instructor_client=instructor_client,
        prompt=prompt_definition,
        class_dict=class_dict,
        model="gpt-4o-mini",
        zenbase_tracer=zenbase_tracer,
        training_set=train_set,
        validation_set=validation_set,
        test_set=test_set,
    )


@pytest.mark.helpers
def test_single_class_classifier_perform(single_class_classifier: SingleClassClassifier, sample_news_article):
    result = single_class_classifier.optimize()
    assert all(
        [result.best_function, result.candidate_results, result.best_candidate_result]
    ), "Assertions failed for result properties"
    traces = single_class_classifier.zenbase_tracer.all_traces
    assert traces, "No traces found"
    assert result is not None, "Result should not be None"
    assert hasattr(result, "best_function"), "Result should have a best_function attribute"
    best_fn = result.best_function
    assert callable(best_fn), "best_function should be callable"
    output = best_fn(sample_news_article)
    assert output is not None, "output should not be None"
