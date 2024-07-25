import csv
import io
from typing import Dict, List

from instructor import Instructor
from pydantic import BaseModel, Field


class SingleClassClassifierSyntheticDataExample(BaseModel):
    inputs: str = Field(..., description="The input text for single class classification")
    outputs: str = Field(..., description="The correct classification category")


class SingleClassClassifierSyntheticDataGenerator:
    def __init__(
        self,
        instructor_client: Instructor,
        prompt: str,
        class_dict: Dict[str, str],
        model: str = "gpt-4o-mini",
    ):
        self.instructor_client = instructor_client
        self.prompt = prompt
        self.class_dict = class_dict
        self.model = model

    def generate_examples_for_category(
        self, category: str, description: str, num_examples: int
    ) -> List[SingleClassClassifierSyntheticDataExample]:
        messages = [
            {
                "role": "system",
                "content": f"""You are an expert in generating synthetic datasets for single class classification
                 tasks. Your task is to create diverse and realistic examples based on the following instructions:

{self.prompt}

You are focusing on generating examples for the following category:
- {category}: {description}

For each example, generate:
1. A realistic and diverse input text that should be classified into the given category.
2. The category name as the output.

Ensure diversity in the generated examples.""",
            },
            {"role": "user", "content": f"Generate {num_examples} examples for the category '{category}'."},
        ]

        response = self.instructor_client.chat.completions.create(
            model=self.model, response_model=List[SingleClassClassifierSyntheticDataExample], messages=messages
        )

        return response

    def generate_examples(self, examples_per_category: int) -> List[SingleClassClassifierSyntheticDataExample]:
        all_examples = []
        for category, description in self.class_dict.items():
            category_examples = self.generate_examples_for_category(category, description, examples_per_category)
            all_examples.extend(category_examples)
        return all_examples

    def generate_csv(self, examples_per_category: int) -> str:
        examples = self.generate_examples(examples_per_category)

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["inputs", "outputs"])
        writer.writeheader()
        for example in examples:
            writer.writerow(example.dict())

        return output.getvalue()

    def save_csv(self, filename: str, examples_per_category: int):
        csv_content = self.generate_csv(examples_per_category)
        with open(filename, "w", newline="", encoding="utf-8") as f:
            f.write(csv_content)
