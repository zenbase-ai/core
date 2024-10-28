import csv
import io
from unittest.mock import Mock, patch

import instructor
import pytest
from openai import OpenAI

from zenbase.predefined.syntethic_data.single_class_classifier import (
    SingleClassClassifierSyntheticDataExample,
    SingleClassClassifierSyntheticDataGenerator,
)


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
    }


@pytest.fixture(scope="module")
def openai_client() -> OpenAI:
    return OpenAI()


@pytest.fixture(scope="module")
def instructor_client(openai_client: OpenAI) -> instructor.Instructor:
    return instructor.from_openai(openai_client)


@pytest.fixture(scope="module")
def synthetic_data_generator(
    instructor_client: instructor.Instructor,
    prompt_definition: str,
    class_dict: dict[str, str],
) -> SingleClassClassifierSyntheticDataGenerator:
    return SingleClassClassifierSyntheticDataGenerator(
        instructor_client=instructor_client,
        prompt=prompt_definition,
        class_dict=class_dict,
        model="gpt-4o-mini",
    )


@pytest.mark.helpers
def test_synthetic_data_generator_initialization(
    synthetic_data_generator: SingleClassClassifierSyntheticDataGenerator,
):
    assert synthetic_data_generator is not None
    assert synthetic_data_generator.instructor_client is not None
    assert synthetic_data_generator.prompt is not None
    assert synthetic_data_generator.class_dict is not None
    assert synthetic_data_generator.model is not None


@pytest.mark.helpers
def test_generate_examples_for_category(
    synthetic_data_generator: SingleClassClassifierSyntheticDataGenerator,
):
    category = "Automobiles"
    description = synthetic_data_generator.class_dict[category]
    num_examples = 5
    examples = synthetic_data_generator.generate_examples_for_category(category, description, num_examples)

    assert len(examples) == num_examples
    for example in examples:
        assert example.inputs is not None
        assert example.outputs == category


@pytest.mark.helpers
def test_generate_examples(
    synthetic_data_generator: SingleClassClassifierSyntheticDataGenerator,
):
    examples_per_category = 3
    all_examples = synthetic_data_generator.generate_examples(examples_per_category)

    assert len(all_examples) == examples_per_category * len(synthetic_data_generator.class_dict)
    for example in all_examples:
        assert example.inputs is not None
        assert example.outputs in synthetic_data_generator.class_dict.keys()


@pytest.mark.helpers
def test_generate_csv(
    synthetic_data_generator: SingleClassClassifierSyntheticDataGenerator,
):
    examples_per_category = 2
    csv_content = synthetic_data_generator.generate_csv(examples_per_category)

    csv_reader = csv.DictReader(io.StringIO(csv_content))
    rows = list(csv_reader)

    assert len(rows) == examples_per_category * len(synthetic_data_generator.class_dict)
    for row in rows:
        assert "inputs" in row
        assert "outputs" in row
        assert row["outputs"] in synthetic_data_generator.class_dict.keys()


@pytest.mark.helpers
def test_save_csv(
    synthetic_data_generator: SingleClassClassifierSyntheticDataGenerator,
    tmp_path,
):
    examples_per_category = 2
    file_path = tmp_path / "test_synthetic_data.csv"
    synthetic_data_generator.save_csv(str(file_path), examples_per_category)

    assert file_path.exists()

    with open(file_path, "r", newline="", encoding="utf-8") as f:
        csv_reader = csv.DictReader(f)
        rows = list(csv_reader)

    assert len(rows) == examples_per_category * len(synthetic_data_generator.class_dict)
    for row in rows:
        assert "inputs" in row
        assert "outputs" in row
        assert row["outputs"] in synthetic_data_generator.class_dict.keys()


@pytest.fixture
def mock_openai_client():
    mock_client = Mock(spec=OpenAI)
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock()
    return mock_client


@pytest.fixture
def mock_instructor_client(mock_openai_client):
    return instructor.from_openai(mock_openai_client)


@pytest.fixture
def mock_generator(mock_instructor_client, class_dict):
    return SingleClassClassifierSyntheticDataGenerator(
        instructor_client=mock_instructor_client,
        prompt="Classify the given text into one of the categories",
        class_dict=class_dict,
        model="gpt-4o-mini",
    )


def mock_generate_examples(
    category: str, description: str, num: int
) -> list[SingleClassClassifierSyntheticDataExample]:
    return [
        SingleClassClassifierSyntheticDataExample(
            inputs=f"Sample text for {category} {i}: {description[:20]}...", outputs=category
        )
        for i in range(num)
    ]


def test_generate_csv_mock(mock_generator):
    examples_per_category = 2

    with patch.object(mock_generator, "generate_examples_for_category", side_effect=mock_generate_examples):
        csv_content = mock_generator.generate_csv(examples_per_category)

        # Parse the CSV content
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)

        assert len(rows) == len(mock_generator.class_dict) * examples_per_category
        for row in rows:
            assert "inputs" in row
            assert "outputs" in row
            assert row["outputs"] in mock_generator.class_dict


def test_save_csv_mock(mock_generator, tmp_path):
    examples_per_category = 2
    filename = tmp_path / "test_output.csv"

    with patch.object(mock_generator, "generate_examples_for_category", side_effect=mock_generate_examples):
        mock_generator.save_csv(str(filename), examples_per_category)

        assert filename.exists()

        with open(filename, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            assert len(rows) == len(mock_generator.class_dict) * examples_per_category
            for row in rows:
                assert "inputs" in row
                assert "outputs" in row
                assert row["outputs"] in mock_generator.class_dict


def test_integration_mock(mock_generator):
    examples_per_category = 1

    def mock_create(**kwargs):
        category = kwargs["messages"][1]["content"].split("'")[1]
        description = next(desc for cat, desc in mock_generator.class_dict.items() if cat == category)
        return mock_generate_examples(category, description, examples_per_category)

    with patch.object(mock_generator.instructor_client.chat.completions, "create", side_effect=mock_create):
        csv_content = mock_generator.generate_csv(examples_per_category)

        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)

        assert len(rows) == len(mock_generator.class_dict) * examples_per_category
        for row in rows:
            assert row["outputs"] in mock_generator.class_dict
            assert row["inputs"].startswith(f"Sample text for {row['outputs']}")
