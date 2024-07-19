import instructor
import pytest
from datasets import DatasetDict
from instructor.client import Instructor
from openai import OpenAI
from pydantic import BaseModel

from zenbase.core.managers import ZenbaseTracer
from zenbase.predefined.single_class_classifier import SingleClassClassifier
from zenbase.predefined.single_class_classifier.function_generator import SingleClassClassifierLMFunctionGenerator

TRAINSET_SIZE = 10
VALIDATIONSET_SIZE = 2
TESTSET_SIZE = 2


@pytest.fixture(scope="module")
def prompt_definition() -> str:
    return """Your task is to accurately categorize each incoming arXiv paper into one of the given categories based
    on its title and abstract."""


@pytest.fixture(scope="module")
def class_dict() -> dict[str, str]:
    return {
        "Machine Learning": "Papers focused on algorithms and statistical models that enable computer systems to "
        "improve their performance on a specific task over time.",
        "Artificial Intelligence": "Research on creating intelligent machines that work and react like humans.",
        "Computational Linguistics": "Studies involving computer processing of human languages.",
        "Information Retrieval": "The science of searching for information in documents, databases, and on the World "
        "Wide Web.",
        "Computer Vision": "Field of study focused on how computers can be made to gain high-level understanding from "
        "digital images or videos.",
        "Human-Computer Interaction": "Research on the design and use of computer technology, focused on the "
        "interfaces between people and computers.",
        "Cryptography and Security": "Studies on secure communication techniques and cybersecurity measures.",
        "Robotics": "Research on the design, construction, operation, and use of robots.",
        "Computers and Society": "Exploration of the social impact of computers and computation on society.",
        "Software Engineering": "Application of engineering to the development of software in a systematic method.",
    }


@pytest.fixture(scope="module")
def sample_arxiv_paper():
    return """title: A Survey of Temporal Credit Assignment in Deep Reinforcement Learning
                abstract: The Credit Assignment Problem (CAP) refers to the longstanding challenge of
                Reinforcement Learning (RL) agents to associate actions with their long-term
                consequences. Solving the CAP is a crucial step towards the successful
                deployment of RL in the real world since most decision problems provide
                feedback that is noisy, delayed, and with little or no information about the
                causes. These conditions make it hard to distinguish serendipitous outcomes
                from those caused by informed decision-making. However, the mathematical nature
                of credit and the CAP remains poorly understood and defined. In this survey, we
                review the state of the art of Temporal Credit Assignment (CA) in deep RL. We
                propose a unifying formalism for credit that enables equitable comparisons of
                state of the art algorithms and improves our understanding of the trade-offs
                between the various methods. We cast the CAP as the problem of learning the
                influence of an action over an outcome from a finite amount of experience. We
                discuss the challenges posed by delayed effects, transpositions, and a lack of
                action influence, and analyse how existing methods aim to address them.
                Finally, we survey the protocols to evaluate a credit assignment method, and
                suggest ways to diagnoses the sources of struggle for different credit
                assignment methods. Overall, this survey provides an overview of the field for
                new-entry practitioners and researchers, it offers a coherent perspective for
                scholars looking to expedite the starting stages of a new study on the CAP, and
                it suggests potential directions for future research"""


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
        model="gpt-3.5-turbo",
        zenbase_tracer=zenbase_tracer,
    )


def test_single_class_classifier_lm_function_generator_initialization(
    single_class_classifier_generator: SingleClassClassifierLMFunctionGenerator,
):
    assert single_class_classifier_generator is not None
    assert single_class_classifier_generator.instructor_client is not None
    assert single_class_classifier_generator.prompt is not None
    assert single_class_classifier_generator.class_dict is not None
    assert single_class_classifier_generator.model is not None
    assert single_class_classifier_generator.zenbase_tracer is not None

    # Check generated class enum and prediction class
    assert single_class_classifier_generator.class_enum is not None
    assert issubclass(single_class_classifier_generator.prediction_class, BaseModel)


@pytest.mark.helpers
def test_single_class_classifier_lm_function_generator_prediction(
    single_class_classifier_generator: SingleClassClassifierLMFunctionGenerator, sample_arxiv_paper
):
    result = single_class_classifier_generator.generate()(sample_arxiv_paper)

    assert result.class_label.name == "Machine Learning"
    assert single_class_classifier_generator.zenbase_tracer.all_traces is not None


@pytest.mark.helpers
def test_single_class_classifier_lm_function_generator_with_missing_data(
    single_class_classifier_generator: SingleClassClassifierLMFunctionGenerator,
):
    faulty_email = {"subject": "Meeting Reminder"}
    result = single_class_classifier_generator.generate()(faulty_email)

    assert result.class_label.name in {
        "Machine Learning",
        "Artificial Intelligence",
        "Computational Linguistics",
        "Information Retrieval",
        "Computer Vision",
        "Human-Computer Interaction",
        "Cryptography and Security",
        "Robotics",
        "Computers and Society",
        "Software Engineering",
    }


def create_dataset_with_examples(item_set: list):
    return [{"inputs": item["input"], "outputs": item["output"]} for item in item_set]


@pytest.fixture(scope="module")
def train_set(arxiv_dataset: DatasetDict):
    return create_dataset_with_examples(
        list(arxiv_dataset["train"].select(range(TRAINSET_SIZE))),
    )


@pytest.fixture(scope="module")
def validation_set(arxiv_dataset: DatasetDict):
    return create_dataset_with_examples(
        list(arxiv_dataset["train"].select(range(TRAINSET_SIZE + 1, TRAINSET_SIZE + VALIDATIONSET_SIZE + 1))),
    )


@pytest.fixture(scope="module")
def test_set(arxiv_dataset: DatasetDict):
    return create_dataset_with_examples(
        list(arxiv_dataset["test"].select(range(TESTSET_SIZE))),
    )


@pytest.fixture(scope="module")
def single_class_classifier(
    instructor_client: Instructor,
    prompt_definition: str,
    class_dict: dict[str, str],
    zenbase_tracer: ZenbaseTracer,
    train_set,
    validation_set,
    test_set,
) -> SingleClassClassifier:
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
def test_single_class_classifier_perform(single_class_classifier: SingleClassClassifier):
    result = single_class_classifier.perform()
    assert result.best_function is not None
    assert result.candidate_results is not None
    assert result.best_candidate_result is not None
    assert single_class_classifier.zenbase_tracer.all_traces is not None
    assert len(single_class_classifier.zenbase_tracer.all_traces) > 0
